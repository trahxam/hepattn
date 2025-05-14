import awkward as ak
import numba
import numpy as np
import torch

from lightning import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from torch import Tensor
from pathlib import Path


@numba.njit
def fill_charge_matrices(matrices, indices, values):
    # For each ROI in the batch
    for i in range(len(matrices)):
        # For each pixel cluster in the ROI
        for j in range(len(indices[i])):
            # For each charge deposit in the cluster
            for k in range(len(indices[i][j])):
                matrices[i][j][indices[i][j][k]] = values[i][j][k]
    return matrices


@numba.njit()
def build_track_hit_masks_bcodes(tracks_bcode, hits_bcodes, track_padding, hit_padding):
    num_rois = len(tracks_bcode)
    track_hit_bcode = np.zeros(shape=(num_rois, track_padding, hit_padding), dtype=np.bool_)
    for roi_idx in range(num_rois):
        for track_idx, track_bcode in enumerate(tracks_bcode[roi_idx]):
            for hit_idx, hit_bcodes in enumerate(hits_bcodes[roi_idx]):
                for hit_bcode in hit_bcodes:
                    if hit_bcode == track_bcode:
                        track_hit_bcode[roi_idx,track_idx,hit_idx] = True

    return track_hit_bcode


@numba.njit()
def build_track_hit_masks_ids(tracks_hits_ids, hits_id, track_padding, hit_padding):
    num_rois = len(tracks_hits_ids)
    track_hit_masks = np.zeros(shape=(num_rois, track_padding, hit_padding), dtype=np.bool_)

    for roi_idx in range(num_rois):
        for track_idx, track_hits_ids in enumerate(tracks_hits_ids[roi_idx]):
            for hit_idx, hit_id in enumerate(hits_id[roi_idx]):
                for track_hit_id in track_hits_ids:
                    if track_hit_id == hit_id:
                        track_hit_masks[roi_idx,track_idx,hit_idx] = True

    return track_hit_masks


# TODO: Wrap this somehow to return a dict instead / generally clean it up
@numba.njit()
def build_track_hit_targets(tracks_bcode, hits_bcodes, hit_phi, hit_theta, hit_locx, hit_locy, hit_energy, track_padding, hit_padding):
    num_rois = len(tracks_bcode)
    track_hit_phi = np.zeros(shape=(num_rois, track_padding, hit_padding), dtype=np.float32)
    track_hit_theta = np.zeros(shape=(num_rois, track_padding, hit_padding), dtype=np.float32)
    track_hit_locx = np.zeros(shape=(num_rois, track_padding, hit_padding), dtype=np.float32)
    track_hit_locy = np.zeros(shape=(num_rois, track_padding, hit_padding), dtype=np.float32)
    track_hit_energy = np.zeros(shape=(num_rois, track_padding, hit_padding), dtype=np.float32)

    for roi_idx in range(num_rois):
        for track_idx, track_bcode in enumerate(tracks_bcode[roi_idx]):
            for hit_idx, hit_bcodes in enumerate(hits_bcodes[roi_idx]):
                for bcode_idx, hit_bcode in enumerate(hit_bcodes):
                    if hit_bcode == track_bcode:
                        track_hit_locx[roi_idx,track_idx,hit_idx] = hit_locx[roi_idx][hit_idx][bcode_idx]
                        track_hit_locy[roi_idx,track_idx,hit_idx] = hit_locy[roi_idx][hit_idx][bcode_idx]
                        track_hit_phi[roi_idx,track_idx,hit_idx] = hit_phi[roi_idx][hit_idx][bcode_idx]
                        track_hit_theta[roi_idx,track_idx,hit_idx] = hit_theta[roi_idx][hit_idx][bcode_idx]
                        track_hit_energy[roi_idx,track_idx,hit_idx] = hit_energy[roi_idx][hit_idx][bcode_idx]
    
    return track_hit_phi, track_hit_theta, track_hit_locx, track_hit_locy, track_hit_energy


def prep_field(x: ak.Array, padding_size: int, fill_value: float):
    """ Takes an input jagged awkward array and converts it to
    a rectangular numpy array that is padded with a given padding value.

    Parameters
    ----------
    x : ak.Array
        Input awkward array.
    padding_size : int
        Size to pad the output to along the jagged dimension.
    fill_value : float
        Vaue to fill padded entries with.

    Returns
    -------
    np.ndarray
        Rectangular array paddded to the given size.
    """
    if x.ndim == 1:
        return ak.to_numpy(x)
    x = ak.pad_none(x, target=padding_size, axis=1)
    if x.ndim == 3:
        fill_value = ak.full_like(x[0][0], fill_value)
    return ak.to_numpy(ak.fill_none(x, value=fill_value, axis=1))


def wrap_phi(x: Tensor) -> Tensor:
    """ Correctly wraps an input tensor of pi angles so they lie in [0, 2pi]. """
    x = x + ak.values_astype(x < -np.pi, np.float32) * 2 * np.pi
    x = x - ak.values_astype(x > np.pi, np.float32) * 2 * np.pi
    return x


class ROIDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        num_samples: int,
        inputs: dict,
        targets: dict,
        sampling_seed: int = 42,
        min_track_pt: float = 1.0,
        min_track_hits: dict[str, int] = {"pix": 2, "sct": 6},
        max_roi_eta: float = 4,
        max_roi_energy: float = 1e9,
        allow_noise_hits: bool = True,
        allow_shared_hits: bool = True,
        max_roi_track_deta: float = 0.05,
        max_roi_track_dphi: float = 0.05,
        max_roi_track_dz0: float = 0.01,
        max_track_d0: float = 0.0025,
        max_roi_hit_deta: float = 0.5,
        max_roi_hit_dphi: float = 0.5,
        only_keep_perfect_rois: bool = True,
    ):
        """
        A dataset which loads and stores jagged unstructured ROI data in memory and dynamically
        converts it to a structured rectangular format on demand.

        Parameters
        ----------
        dirpath : str
            Path to directory containing preprocessed awkward arrays which contain ROI data.
        num_samples : int
            Number of samples to retain after cuts are applied.
        features : dict
            Dictionary whose keys represent the feature name, e.g a pixel cluster, which has three entries:
                - pad : int, size of padding to pad the features to
                - min : int, minimum number of the features in an ROI, e.g. the minium number of pixel clusters to allow
                - max : int, maximum number of features in the ROI, must be less than or equal to the pad value
        sampling seed : int
            Global seed for all random sampling.
        min_track_pt : float
            Minimum pT in GeV for a track to be deemed reconstructable.
        min_track_pt : dict[str, int]
            Dictionary specifying minimum number of pixel / sct hits required for a track to be deemed reconstructable.
        max_roi_eta : float
            Maximum ROI axis eta for ROI to be retained.
        max_roi_energy : float
            Maximum ROI energy in GeV for ROI to be retained.
        allow_noise_hits : bool
            Whether to retain hits that do not belong to a reconstructable track.
        allow_noise_hits : bool
            Whether to retain hits that are shared between multiple reconstructable tracks.
        max_roi_track_deta : float
            Maximum angular eta distance between a track and ROI axis for a track to be deemed reconstructable.
        max_roi_track_dphi : float
            Maximum angular phi distance between a track and ROI axis for a track to be deemed reconstructable.
        max_roi_track_dz0 : float
            Maximum distance between a track and ROI axis z origin in mm for a track to be deemed reconstructable.
        max_track_d0 : float
            Maximum track d0 for a track to be deemed reconstructable.
        max_roi_hit_deta : float
            Maximum angular eta distance between a hit and ROI axis for a hit to be marked as valid.
        max_roi_hit_dphi : float
            Maximum angular phi distance between a hit and ROI axis for a hit to be marked as valid.
        only_keep_perfect_rois : bool
            If true, only ROIs for which all tracks passed the given track cuts will be retained.
        """
        super().__init__()

        self.sampling_seed = sampling_seed
        self.dirpath = Path(dirpath)
        self.num_samples = num_samples
        self.inputs = inputs
        self.targets = targets
        self.features = inputs | targets
        self.min_track_pt = min_track_pt
        self.min_track_hits = min_track_hits
        self.max_roi_eta = max_roi_eta
        self.max_roi_energy = max_roi_energy
        self.allow_noise_hits = allow_noise_hits
        self.allow_shared_hits = allow_shared_hits
        self.max_roi_track_deta = max_roi_track_deta
        self.max_roi_track_dphi = max_roi_track_dphi
        self.max_roi_track_dz0 = max_roi_track_dz0
        self.max_track_d0 = max_track_d0
        self.max_roi_hit_deta = max_roi_hit_deta
        self.max_roi_hit_dphi = max_roi_hit_dphi
        self.only_keep_perfect_rois = only_keep_perfect_rois

        self.dirpath = Path(self.dirpath)

        np.random.seed(self.sampling_seed)

        data = []
        num_samples_loaded = 0
        mem_usage_gb = 0

        print("=" * 100 + f"\nReading files from {self.dirpath}\n" + "=" * 100)
        for file_path in self.dirpath.iterdir():
            if not file_path.suffix == ".parquet":
                continue
            print(f"Loading {file_path}")
            x = ak.from_parquet(file_path)
            pre_cut_size = len(x)
            print(f"Loaded {pre_cut_size} ROIs from {file_path}")

            # Remove ROIs that have two many/few of each of the features
            for feature, info in self.features.items():
                if feature == "roi":
                    continue
                x[f"roi_num_{feature}"] = ak.num(x[feature + "_id"])
                x = x[x[f"roi_num_{feature}"] <= info["max"]]
                x = x[x[f"roi_num_{feature}"] >= info["min"]]
                assert info["max"] <= info["pad"], f"Padding must be >= max count for {feature}"
            
            # ROI level cuts
            x = x[np.abs(x[f"roi_eta"]) <= self.max_roi_eta]
            x = x[x[f"roi_e"] <= self.max_roi_energy]

            x_tmp = {}

            roi_eta = x["roi_eta"]
            roi_phi = x["roi_phi"]

            # Calculate a dirty vertex to use as the ROI reference z
            roi_z0 = ak.mean(x[f"sisp_z0"], axis=-1)

            sisp_z0 = prep_field(x[f"sisp_z0"], self.features["sisp"]["pad"], np.nan)
            roi_z0 = np.nanmedian(sisp_z0, axis=-1)
            roi_z0 = ak.from_numpy(roi_z0)

            # Apply hit cuts
            for k in ["pix", "sct"]:
                x_tmp[f"{k}_valid"] = prep_field(ak.full_like(x[f"{k}_id"], True, dtype=bool), self.features[k]["pad"], False)

                # Note we use the eta calculated using the ROI reference z
                k_cor_z = x[f"{k}_z"] - roi_z0
                k_s = np.sqrt(x[f"{k}_x"] ** 2 + x[f"{k}_y"] ** 2 + k_cor_z ** 2)
                k_theta = np.arccos(k_cor_z / k_s)
                k_eta = -np.log(np.tan(k_theta / 2))
                k_phi = np.arctan2(x[f"{k}_y"], x[f"{k}_x"])

                roi_k_deta = np.abs(k_eta - roi_eta) 
                roi_k_dphi = np.abs(wrap_phi(k_phi - roi_phi))

                # Apply any hit-based cuts by marking hits that fail the cuts as invalid hit slots
                x_tmp[f"{k}_valid"] = x_tmp[f"{k}_valid"] & prep_field(roi_k_deta <= self.max_roi_hit_deta, self.features[k]["pad"], False)
                x_tmp[f"{k}_valid"] = x_tmp[f"{k}_valid"] & prep_field(roi_k_dphi <= self.max_roi_hit_dphi, self.features[k]["pad"], False)

            sudo_initial_valid = prep_field((x[f"sudo_pt"]) >= 0.0, self.features["sudo"]["pad"], False)

            # Apply track cuts
            for q in ["sudo", "sisp", "reco"]:
                print(f"\nPerforming track paramater cuts for {q}")

                # ROI-track angle containment cut
                q_eta = x[f"{q}_eta"]
                q_phi = x[f"{q}_phi"]

                roi_q_abs_deta = np.abs(q_eta - roi_eta) 
                roi_q_abs_dphi = np.abs(wrap_phi(q_phi - roi_phi))

                # Scale d0 from mm -> m
                roi_q_abs_dz0 = 0.001 * np.abs(x[f"{q}_z0"] - roi_z0)
                q_abs_d0 = 0.001 * np.abs(x[f"{q}_d0"])

                # Scale track pT from MeV to GeV
                q_pt = x[f"{q}_pt"] / 1000.0

                # Apply any track-based cuts by marking tracks that fail the cuts as invalid track slots
                track_cuts = {
                    f"track pt >= {self.min_track_pt}":  q_pt >= self.min_track_pt,
                    f"roi-track deta <= {self.max_roi_track_deta}": roi_q_abs_deta <= self.max_roi_track_deta,
                    f"roi-track dphi <= {self.max_roi_track_dphi}": roi_q_abs_dphi <= self.max_roi_track_dphi,
                    f"roi-track dz0 <= {self.max_roi_track_dz0}": roi_q_abs_dz0 <= self.max_roi_track_dz0,
                    f"track d0 <= {self.max_track_d0}": q_abs_d0 <= self.max_track_d0,
                }

                # Initialise vlid tracks
                initial_valid = prep_field((x[f"{q}_pt"]) >= 0.0, self.features[q]["pad"], False)
                x_tmp[f"{q}_valid"] = initial_valid.copy()

                for cut_name, cut_mask in track_cuts.items():
                    pre_cut_num = x_tmp[f"{q}_valid"].sum()
                    x_tmp[f"{q}_valid"] = x_tmp[f"{q}_valid"] & prep_field(cut_mask, self.features[q]["pad"], False)
                    post_cut_num = x_tmp[f"{q}_valid"].sum()
                    retenetion_rate = 100 * post_cut_num / pre_cut_num
                    print(f"Applied track cut {cut_name}: {pre_cut_num} pre-cut, {post_cut_num} post-cut ({retenetion_rate:.2f})")

            # Build track-hit masks so we can do noise/sharing cuts
            for q in ["sudo", "sisp", "reco"]:
                for k in ["pix", "sct"]:
                    if q == "sudo":
                        track_hit_masks = build_track_hit_masks_bcodes(x[f"{q}_bcode"], x[f"{k}_bcodes"], self.features[q]["pad"], self.features[k]["pad"])
                    else:
                        track_hit_masks = build_track_hit_masks_ids(x[f"{q}_clus_ids"], x[f"{k}_id"], self.features[q]["pad"], self.features[k]["pad"])
                    x_tmp[f"{q}_{k}_masks"] = track_hit_masks
                    x_tmp[f"{q}_{k}_masks"] = x_tmp[f"{q}_{k}_masks"] & x_tmp[f"{q}_valid"][:,:,None]
                    x_tmp[f"{q}_{k}_masks"] = x_tmp[f"{q}_{k}_masks"] & x_tmp[f"{k}_valid"][:,None,:]

            # Apply hit cuts that depend on other hits/tracks, i.e. noise and shared hit cuts
            for k in ["pix", "sct"]:
                # Removing noise hits simply removes / marks as null any hits that are not associated to a pseudotrack / particle
                if not self.allow_noise_hits:
                    x_tmp[f"{k}_valid"] = x_tmp[f"{k}_valid"] & (x_tmp[f"sudo_{k}_masks"].sum(-2) > 0)

                for q in ["sudo", "sisp", "reco"]:
                    print(f"\nPerforming track hit content cuts for {q}")
                    # Removing shared hits will remove any tracks that are associated to any shared hit
                    # and also any hits that are associated to tracks on a shared hit
                    if not self.allow_shared_hits:
                        shared_hit = x_tmp[f"sudo_{k}_masks"].sum(-2) > 1 # B x num_hits
                        track_with_shared_hit = (x_tmp[f"{q}_{k}_masks"] & shared_hit[:,None,:]).any(-1) # B x num_tracks
                        hit_on_shared_track = (x_tmp[f"{q}_{k}_masks"] & track_with_shared_hit[:,:,None]).any(-2)
                        x_tmp[f"{k}_valid"] = x_tmp[f"{k}_valid"] & (~hit_on_shared_track)
                        x_tmp[f"{q}_valid"] = x_tmp[f"{q}_valid"] & (~track_with_shared_hit)

                    x_tmp[f"{q}_{k}_masks"] = x_tmp[f"{q}_{k}_masks"] & x_tmp[f"{q}_valid"][:,:,None]
                    x_tmp[f"{q}_{k}_masks"] = x_tmp[f"{q}_{k}_masks"] & x_tmp[f"{k}_valid"][:,None,:]

                    pre_cut_num = x_tmp[f"{q}_valid"].sum()
                    x_tmp[f"{q}_valid"] = x_tmp[f"{q}_valid"] & (x_tmp[f"{q}_{k}_masks"].sum(-1) >= self.min_track_hits[k])
                    post_cut_num = x_tmp[f"{q}_valid"].sum()
                    retenetion_rate = 100 * post_cut_num / pre_cut_num
                    print(f"Applied track cut num {k} >= {self.min_track_hits[k]}: {pre_cut_num} pre-cut, {post_cut_num} post-cut ({retenetion_rate:.2f})")
            
            # Save which hits and tracks are valid after we applied cuts
            for q in ["sudo", "sisp", "reco"]:
                x[f"roi_num_{q}"] = x_tmp[f"{q}_valid"].sum(-1)
                x[f"{q}_valid"] = x_tmp[f"{q}_valid"]
                
            for k in ["pix", "sct"]:
                x[f"roi_num_{k}"] = x_tmp[f"{k}_valid"].sum(-1)
                x[f"{k}_valid"] = x_tmp[f"{k}_valid"]
            
            # Mark ROIs that did not loose any tracks during the cut selection
            x["roi_no_sudo_dropped"] = ak.from_numpy(np.all(x_tmp[f"sudo_valid"] == sudo_initial_valid, axis=-1))

            # Apply ROI-level cuts
            x = x[x["roi_num_sudo"] >= 1]

            if self.only_keep_perfect_rois:
                x = x[x["roi_no_sudo_dropped"]]

            post_cut_size = len(x)
            data.append(x)
            num_samples_loaded += post_cut_size
            mem_usage_gb += x.nbytes / (1024 * 1024 * 1024)
            pct_loaded = 100 * num_samples_loaded / num_samples
            print(f"\nRead {pre_cut_size}, retained {post_cut_size} now have " +
                  f"{num_samples_loaded} of {num_samples} target ({pct_loaded:.2f}%), using {mem_usage_gb:.2f}GB\n")

            # If we have already loaded enough clusters then dont read any more files
            if num_samples_loaded >= num_samples and num_samples != -1:
                break

        # Concatenate the data from all the files and get rid of any excess samples we loaded
        print(f"Done loading data, concatenating")
        data = ak.concatenate(data, axis=0)
        if num_samples_loaded > num_samples and num_samples != -1:
            data = data[:num_samples]

        self.data = data
        self.num_samples = len(self.data)
        print(f"After cuts and skimming, have {self.num_samples} samples")
        print(f"Final dataset type {data.type}")

    def __len__(self) -> int:
        """ Returns the number of samples / ROIs that are available in the dataset after all cuts have been applied. """
        return int(self.num_samples)
    
    def __getitem__(self, idx) -> dict[str, Tensor]:
        """ Return a minibatch of data at the given indices.

        Parameters
        ----------
        idx : slice
            An array slice specifying which data from the overall dataset in memory to return.

        Returns
        -------
        data : dict[str, Tensor]
            A dictionary of tensors containing data for the minibatch. Each key specifies the attribute name, and each
            value contains a tensor with the values for the field. For example: "pix_x" contains a float tensor of shape
            (batch size, pixel padding size) which contains the x-coordinates of the pixel clusters in the mini batch.
        """
        batch = self.data[idx]

        # Additional ROI axis coords and log of ROI energy 
        batch["roi_theta"] = 2 * np.arctan(np.exp(-batch["roi_eta"]))
        batch["roi_loge"] = np.log(batch["roi_e"] + 1.0)

        # Convert the track z0 and d0 from mm to m
        for q in ["sudo", "sisp", "reco"]:
            batch[f"{q}_z0"] = 0.001 * batch[f"{q}_z0"]
            batch[f"{q}_d0"] = 0.001 * batch[f"{q}_d0"]

        # Calculate a rough ROI 'vertex' by taking the median z0 of the sisp tracks in the ROI
        sisp_z0 = prep_field(batch[f"sisp_z0"], self.features["sisp"]["pad"], np.nan)
        roi_z0 = np.nanmedian(sisp_z0, axis=-1)
        batch["roi_z0"] = ak.from_numpy(roi_z0)
        
        # 7x7 charge matrix for the pixel clusters - use logarithm due to large dynamic range of deposited charge
        chargemats = np.zeros(shape=(len(batch), self.features["pix"]["pad"], 49))
        chargemats = fill_charge_matrices(chargemats, batch["pix_chargemat_idx"], batch["pix_chargemat_val"])
        batch["pix_logchargemat"] = np.log(1.0 + chargemats)

        # Calculate cluster cylindrical coordinates and relative ROI coords
        for k in ["pix", "sct"]:
            # Convert mm to m
            # Global cartesian coordinates of the hit
            for i in ["x", "y", "z"]:
                batch[f"{k}_{i}"] = 0.001 * batch[f"{k}_{i}"]
                batch[f"{k}_mod_{i}"] = 0.001 * batch[f"{k}_mod_{i}"]

            # Global cartesian coordinates of the module to which the hit belongs
            for i in ["x", "y"]:
                batch[f"{k}_mod_loc_{i}"] = 0.001 * batch[f"{k}_mod_loc_{i}"]

            # Coordinates in the detector global frame
            batch[f"{k}_r"] = np.sqrt(batch[f"{k}_x"]**2 + batch[f"{k}_y"]**2)
            batch[f"{k}_s"] = np.sqrt(batch[f"{k}_x"]**2 + batch[f"{k}_y"]**2 + batch[f"{k}_z"]**2)
            batch[f"{k}_ryz"] = np.sqrt(batch[f"{k}_y"]**2 + batch[f"{k}_z"]**2)
            batch[f"{k}_absz"] = np.abs(batch[f"{k}_z"])
            batch[f"{k}_u"] = batch[f"{k}_x"] / (batch[f"{k}_x"]**2 + batch[f"{k}_y"]**2)
            batch[f"{k}_v"] = batch[f"{k}_y"] / (batch[f"{k}_x"]**2 + batch[f"{k}_y"]**2)
            batch[f"{k}_theta"] = np.arccos(batch[f"{k}_z"] / batch[f"{k}_s"])
            batch[f"{k}_phi"] = np.arctan2(batch[f"{k}_y"], batch[f"{k}_x"])
            batch[f"{k}_eta"] = -np.log(np.tan(batch[f"{k}_theta"] / 2))
            
            # Calculate module orientation angles
            batch[f"{k}_mod_norm_phi"] = np.arctan2(batch[f"{k}_mod_norm_y"], batch[f"{k}_mod_norm_x"])
            batch[f"{k}_mod_norm_theta"] = np.arccos(batch[f"{k}_mod_norm_z"])

            batch[f"{k}_mod_s"] = np.sqrt(batch[f"{k}_mod_x"]**2 + batch[f"{k}_mod_y"]**2 + batch[f"{k}_mod_z"]**2)
            batch[f"{k}_mod_theta"] = np.arccos(batch[f"{k}_mod_z"] / batch[f"{k}_mod_s"])
            batch[f"{k}_mod_phi"] = np.arctan2(batch[f"{k}_mod_y"], batch[f"{k}_mod_x"])
            batch[f"{k}_mod_eta"] = -np.log(np.tan(batch[f"{k}_mod_theta"] / 2))

            # Calculate a reference z position for the ROI
            batch[f"{k}_cor_z"] = batch[f"{k}_z"] - batch["roi_z0"]
            batch[f"{k}_cor_s"] = np.sqrt(batch[f"{k}_x"]**2 + batch[f"{k}_y"]**2 + batch[f"{k}_cor_z"]**2)
            
            # Calculate hit theta/eta from the ROI 'vertex'
            batch[f"{k}_cor_theta"] =  np.arccos(batch[f"{k}_cor_z"] / batch[f"{k}_cor_s"])
            batch[f"{k}_cor_eta"] = -np.log(np.tan(batch[f"{k}_cor_theta"] / 2))

            # Coordinates relative to the ROI axis
            batch[f"{k}_dtheta"] = batch[f"{k}_cor_theta"] - batch["roi_theta"]
            batch[f"{k}_deta"] = batch[f"{k}_cor_eta"] - batch["roi_eta"]

            # Need to make sure to wrap the global phi angle!
            batch[f"{k}_dphi"] = wrap_phi(batch[f"{k}_phi"] - batch["roi_phi"])
            batch[f"{k}_dR"] = np.sqrt(batch[f"{k}_deta"] ** 2 + batch[f"{k}_dphi"] ** 2)

            # Total charge on the cluster - use the logarithm due to the huge dynamic range of charge
            batch[f"{k}_logcharge"] = np.log(np.maximum(batch[f"{k}_charge"], 0.0) + 1.0)
            
            # Add ROI fields
            for i in ["loge", "phi", "theta", "eta", "m", "z0"]:
                batch[f"{k}_roi_{i}"] = ak.broadcast_arrays(batch[f"roi_{i}"], batch[f"{k}_x"])[0]

        batch["sudo_fromb"] = batch["sudo_bhadpt"] > 0.0

        # Add in additional track fields
        for q in ["sudo", "sisp", "reco"]:
            batch[f"{q}_pt"] = batch[f"{q}_pt"] / 1000
            batch[f"{q}_px"] = batch[f"{q}_pt"] * np.cos(batch[f"{q}_phi"])
            batch[f"{q}_py"] = batch[f"{q}_pt"] * np.sin(batch[f"{q}_phi"])
            batch[f"{q}_pz"] = batch[f"{q}_pt"] * np.sinh(batch[f"{q}_eta"])
            batch[f"{q}_theta"] = 2 * np.arctan(np.exp(-batch[f"{q}_eta"]))
            batch[f"{q}_qop"] = batch[f"{q}_q"] / batch[f"{q}_pt"]

            batch[f"{q}_deta"] = batch[f"{q}_eta"] - batch["roi_eta"]
            batch[f"{q}_dtheta"] = batch[f"{q}_theta"] - batch["roi_theta"]
            batch[f"{q}_dphi"] = wrap_phi(batch[f"{q}_phi"] - batch["roi_phi"])
            batch[f"{q}_dz0"] = batch[f"{q}_z0"] - batch[f"roi_z0"]

        data = {}

        # Now all of the derived fields have been made, prep them into tensors
        for k, v in self.features.items():
            for field in v["fields"]:
                x = prep_field(batch[f"{k}_{field}"], self.features[k]["pad"], 0.0)
                data[f"{k}_{field}"] = torch.from_numpy(x).float()

            # We don't need to construct a valid flag for ROIs - all ROIs retained are valid
            if k == "roi":
                continue

            # The mask that denotes whether an entry is padding or not
            data[f"{k}_valid"] = torch.from_numpy(ak.to_numpy(batch[f"{k}_valid"])).bool()
        
        # Build the track-hit assignment masks
        for k in ["pix", "sct"]:
            for q in ["sudo", "sisp", "reco"]:
                if q == "sudo":
                    x = build_track_hit_masks_bcodes(batch[f"{q}_bcode"], batch[f"{k}_bcodes"], self.features[q]["pad"], self.features[k]["pad"])
                else:
                    x = build_track_hit_masks_ids(batch[f"{q}_clus_ids"], batch[f"{k}_id"], self.features[q]["pad"], self.features[k]["pad"])
                data[f"{q}_{k}_valid"] = torch.from_numpy(x).bool()

        # sudo_pix_targets = build_track_hit_targets(batch["sudo_bcode"], batch["pix_bcodes"], 
        #                                            batch["pix_sudo_loc_phi"], batch["pix_sudo_loc_theta"],
        #                                            batch["pix_sudo_loc_x"], batch["pix_sudo_loc_y"], batch["pix_energydep"], 
        #                                            self.features["sudo"]["pad"], self.features["pix"]["pad"])
        # sudo_pix_phi, sudo_pix_theta, sudo_pix_x, sudo_pix_y, sudo_pix_energy = sudo_pix_targets

        # # Clamp these fields to remove any extreme values that could mess up the regression
        # # Extreme angles can come from e.g. low pt tracks going the other way into the sensor
        # sudo_pix_phi = np.clip(sudo_pix_phi, -0.2, 0.4)
        # sudo_pix_theta = np.clip(sudo_pix_theta, -np.pi/2, np.pi/2)
        # sudo_pix_x = np.clip(sudo_pix_x, -8, 8)
        # sudo_pix_y = np.clip(sudo_pix_y, -6, 6)
        # sudo_pix_energy = np.clip(sudo_pix_energy, 0.0, 0.6)
        
        # # Add the track-hit regression targets
        # # TODO: Make this optional in the config since these use a lot of memory
        # data["sudo_pix_x"] = torch.from_numpy(sudo_pix_x).float()
        # data["sudo_pix_y"] = torch.from_numpy(sudo_pix_y).float()
        # data["sudo_pix_phi"] = torch.from_numpy(sudo_pix_phi).float()
        # data["sudo_pix_theta"] = torch.from_numpy(sudo_pix_theta).float()
        # data["sudo_pix_energy"] = torch.from_numpy(sudo_pix_energy).float()

        # Update the track-hit masks to reflect any hit or track slots that are null
        for k in ["pix", "sct"]:
            for q in ["sudo", "sisp", "reco"]:
                data[f"{q}_{k}_valid"] = data[f"{q}_{k}_valid"] & data[f"{k}_valid"][:,None,:]
                data[f"{q}_valid"] = data[f"{q}_valid"] & (data[f"{q}_{k}_valid"].sum(-1) > 0)
                data[f"{q}_{k}_valid"] = data[f"{q}_{k}_valid"] & data[f"{q}_valid"][:,:,None]

        inputs = {}
        targets = {}
        for k, v in self.inputs.items():
            inputs[f"{k}_valid"] = data[f"{k}_valid"]
            targets[f"{k}_valid"] = data[f"{k}_valid"]
            for field in v["fields"]:
                inputs[f"{k}_{field}"] = data[f"{k}_{field}"]

        for k, v in self.targets.items():
            if k != "roi":
                targets[f"{k}_valid"] = data[f"{k}_valid"]
            for field in v["fields"]:
                targets[f"{k}_{field}"] = data[f"{k}_{field}"]

        for k in ["pix", "sct"]:
            for q in ["sudo", "sisp", "reco"]:
                targets[f"{q}_{k}_valid"] = data[f"{q}_{k}_valid"]

        return inputs, targets


class ROIDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str = None,
        **kwargs,
    ):
        """ Lightning data module. Will iterate over the given directories
        and read preprocessed awkward parquet files until the desired number
        samples are obtained for each dataset.

        Parameters
        ----------
        batch_dimension : int
            Number of samples to read in a minibatch.
        train_dir : str
            Training data directory.
        val_dir : str
            Validation data directory.
        test_dir : str
            Test data directory.
        num_workers : int
            Number of workers / threads too use to read batches.
        num_train " int
            Target number of training samples to load.
        num_val " int
            Target number of training samples to load.
        num_test " int
            Target number of training samples to load.
        """
        super().__init__()

        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.kwargs = kwargs

    def setup(self, stage: str):
        if self.trainer.is_global_zero:
            print("-" * 100)

        # Create training and validation datasets
        if stage == "fit":
            self.train_dset = ROIDataset(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs)
            self.val_dset = ROIDataset(dirpath=self.val_dir, num_samples=self.num_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = ROIDataset(dirpath=self.test_dir, num_samples=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

        if self.trainer.is_global_zero:
            print("-" * 100, "\n")

    def get_dataloader(self, stage: str, dataset: ROIDataset, shuffle: bool):
        random_sampler = RandomSampler(dataset)
        random_batch_sampler = BatchSampler(random_sampler, batch_size=self.batch_size, drop_last=False)
        return DataLoader(dataset=dataset, sampler=random_batch_sampler, batch_size=None)

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=False)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)
