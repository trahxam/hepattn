import random
from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from hepattn.utils.tensor_utils import pad_to_size


class PixelClusterDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_clusters: int = 1000000,
        cluster_multiplicity_sample_frac: dict[int, float] | None = None,
        particle_max_x: float = 10.0,
        particle_max_y: float = 8.0,
        particle_allow_notruth: bool = True,
        particle_allow_secondary: bool = True,
        cluster_regions: list[int] | None = None,
        cluster_layers: list[int] | None = None,
        cluster_min_num_particles=1,
        cluster_max_num_particles=32,
        precision: int = 32,
    ):
        if cluster_layers is None:
            cluster_layers = [0, 1, 2, 4, 5, 6]
        if cluster_regions is None:
            cluster_regions = [-2, -1, 0, 1, 2]
        if cluster_multiplicity_sample_frac is None:
            cluster_multiplicity_sample_frac = {}

        super().__init__()

        self.dirpath = dirpath
        self.inputs = inputs
        self.targets = targets
        self.num_clusters = num_clusters
        self.particle_max_x = particle_max_x
        self.particle_max_y = particle_max_y
        self.particle_allow_notruth = particle_allow_notruth
        self.particle_allow_secondary = particle_allow_secondary
        self.cluster_regions = cluster_regions
        self.cluster_layers = cluster_layers
        self.cluster_multiplicity_sample_frac = cluster_multiplicity_sample_frac
        self.cluster_max_num_particles = cluster_max_num_particles
        self.cluster_min_num_particles = cluster_min_num_particles
        self.precision = precision

        self.precision_type = {
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }[precision]

        # Set the global random sampling seed
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)  # noqa: NPY002

        # Files that are available to read from
        self.available_file_paths = list(Path(self.dirpath).glob("*.h5"))
        print(f"Found {len(self.available_file_paths)} available files in {self.dirpath}")

        # Files that have been registered
        self.file_paths = []

        # Maps the cluster ID to the root file it is in
        self.cluster_id_to_file_path = {}

        # Map the cluster ID to the dataset index and vice-versa
        self.idx_to_cluster_id = {}
        self.cluster_id_to_idx = {}

        # Map the cluster ID to its index in its file
        self.cluster_id_to_file_idx = {}

        # Clusters that are known to fail the selection
        self.rejected_cluster_ids = []
        # Clusters that are known to pass the selection
        self.accepted_cluster_ids = []
        # Clusters that have not yet been evaluated on the selection
        self.unevaluated_cluster_ids = []

        # At the start, we load enough files so that we will have enough for the requested sample size
        # As we read through the dataset, some clusters will be discarded, upon which we will load more files lazily
        for file_path in self.available_file_paths:
            self.register_file(file_path)
            total_num_clusters = len(self.cluster_id_to_file_path)

            if total_num_clusters >= self.num_clusters:
                print(f"Finished registering {total_num_clusters} clusters from {len(self.file_paths)} files")
                break

    def register_file(self, file_path):
        with h5py.File(file_path, "r") as file:
            cluster_ids = file["cluster_id"][:]

            for idx, cluster_id in enumerate(cluster_ids):
                self.cluster_id_to_file_path[cluster_id] = file_path
                self.cluster_id_to_file_idx[cluster_id] = idx
                self.unevaluated_cluster_ids.append(int(cluster_id))

            print(f"Registered {len(cluster_ids)} clusters from {file_path}")
            self.file_paths.append(file_path)
            return

    def __len__(self):
        return int(self.num_clusters)

    def __getitem__(self, idx):
        # Attempt to load the cluster with the given ID
        # First check if an cluster has already been evaluated and assigned to this index
        if idx in self.idx_to_cluster_id:
            # If its been assigned to an index, we know it passes the selection, and so we can go ahead and load it
            cluster = self.load_cluster(self.idx_to_cluster_id[idx])
        else:
            cluster = None

            # Keep trying cluster IDs from the set of unevaluated cluster IDs until we eventually get an cluster that
            # passes the selection
            while cluster is None:
                # Check we still have some clusters left
                if not len(self.unevaluated_cluster_ids) > 0:
                    # If not, we load in more clusters
                    print("Ran out of clusters, so loading new file")
                    unregistered_file_paths = set(self.available_file_paths) - set(self.file_paths)

                    # Check if we have no files left and have ran out of clusters
                    if len(unregistered_file_paths) == 0:
                        raise StopIteration("Ran out of clusters that pass the selection, and have no new files left to read from.")

                    # Load a random How new file then continue
                    self.register_file(next(iter(unregistered_file_paths)))

                # Randomly sample an ID from the set of unevaluated IDs
                cluster_id = random.choice(self.unevaluated_cluster_ids)

                # Load this cluster ID and see if it passes the selection
                cluster = self.load_cluster(cluster_id)

                # This cluster ID has been evaluated on the selection now
                self.unevaluated_cluster_ids.remove(cluster_id)

                # If the cluster passes the selection, add it to the accepted list and map it to a dataset index
                if cluster is not None:
                    self.accepted_cluster_ids.append(cluster_id)
                    self.idx_to_cluster_id[idx] = cluster_id
                    self.cluster_id_to_idx[cluster_id] = idx
                # If the cluster fails the selection, add it to the rejected list and try again
                else:
                    self.rejected_cluster_ids.append(cluster_id)

        # Convert to a torch tensor of the correct dtype and add the batch dimension
        inputs = {}
        targets = {}
        for input_name, fields in self.inputs.items():
            inputs[f"{input_name}_valid"] = torch.from_numpy(cluster[f"{input_name}_valid"]).bool().unsqueeze(0)
            # Some tasks might require to know hit padding info for loss masking
            targets[f"{input_name}_valid"] = inputs[f"{input_name}_valid"]
            for field in fields:
                inputs[f"{input_name}_{field}"] = torch.from_numpy(cluster[f"{input_name}_{field}"]).to(self.precision_type).unsqueeze(0)

        # Convert the targets
        for target_name, fields in self.targets.items():
            targets[f"{target_name}_valid"] = torch.from_numpy(cluster[f"{target_name}_valid"]).bool().unsqueeze(0)
            for field in fields:
                targets[f"{target_name}_{field}"] = torch.from_numpy(cluster[f"{target_name}_{field}"]).to(self.precision_type).unsqueeze(0)

        # Convert the metedata
        targets["sample_id"] = torch.tensor(cluster["cluster_id"], dtype=torch.int64)

        return inputs, targets

    def load_cluster(self, cluster_id):
        file_path = self.cluster_id_to_file_path[cluster_id]
        file_idx = self.cluster_id_to_file_idx[cluster_id]

        x = {"cluster_id": cluster_id}

        with h5py.File(file_path) as file:
            # Apply region cut using barrel endcap flag
            x["cluster_bec"] = file["cluster_bec"][file_idx]
            if x["cluster_bec"] not in self.cluster_regions:
                return None

            # Apply layer cut
            x["cluster_layer"] = file["cluster_layer"][file_idx]
            if x["cluster_layer"] not in self.cluster_layers:
                return None

            # Build the cluster cell coordinates
            x["cluster_eta_index"] = file["cluster_eta_index"][file_idx]
            x["cluster_phi_index"] = file["cluster_phi_index"][file_idx]

            x["cluster_module_eta"] = file["cluster_module_eta"][file_idx]
            x["cluster_module_phi"] = file["cluster_module_phi"][file_idx]

            x["cell_eta_index"] = file["cell_eta_index"][file_idx]
            x["cell_phi_index"] = file["cell_phi_index"][file_idx]

            # Convert the charge to 100 * ke
            x["cell_charge"] = file["cell_charge"][file_idx] / 100000.0

            x["cell_y"] = x["cell_eta_index"] - file["cluster_weighted_eta_index"][file_idx]
            x["cell_x"] = x["cell_phi_index"] - file["cluster_weighted_phi_index"][file_idx]

            x["cluster_size_x"] = file["cluster_size_x"][file_idx]
            x["cluster_size_y"] = file["cluster_size_y"][file_idx]

            x["cluster_width_x"] = np.max(x["cell_x"]) - np.min(x["cell_x"])
            x["cluster_width_y"] = np.max(x["cell_y"]) - np.min(x["cell_y"])

            # Add the charge matrix and pitch vector
            x["cluster_charge_matrix"] = file["cluster_charge_matrix"][file_idx]
            x["cluster_charge_matrix"] = np.flipud(x["cluster_charge_matrix"].reshape(7, 7).T).flatten()
            x["cluster_pitch_vector"] = file["cluster_pitch_vector"][file_idx]

            # Add the particle info
            x["particle_barcode"] = file["particle_barcode"][file_idx]

            x["particle_x"] = file["particle_index_x"][file_idx]
            x["particle_y"] = file["particle_index_y"][file_idx]

            x["particle_theta"] = file["particle_theta"][file_idx]
            x["particle_phi"] = file["particle_phi"][file_idx]

            # Convert MeV to GeV and TeV
            x["particle_p"] = file["particle_p"][file_idx] / 1000.0
            x["particle_p_tev"] = x["particle_p"] / 1000.0

            x["particle_valid"] = x["particle_barcode"] > -1
            x["particle_notruth"] = x["particle_barcode"] <= 0
            x["particle_secondary"] = x["particle_barcode"] >= 200000
            x["particle_primary"] = (x["particle_barcode"] > 0) & (x["particle_barcode"] <= 100000)

            # Apply any particle cuts

            # Ignore particles with no barcode if specified
            if not self.particle_allow_notruth:
                x["particle_valid"] = x["particle_valid"] & (~x["particle_notruth"])

            # Ignore secondary particles if specified
            if not self.particle_allow_secondary:
                x["particle_valid"] = x["particle_valid"] & (~x["particle_secondary"])

            # Apply particle position cuts
            x["particle_valid"] = x["particle_valid"] & (np.abs(x["particle_x"]) <= self.particle_max_x)
            x["particle_valid"] = x["particle_valid"] & (np.abs(x["particle_y"]) <= self.particle_max_y)

            particle_fields = [
                "x",
                "y",
                "theta",
                "phi",
                "p",
                "barcode",
                "p_tev",
                "primary",
                "secondary",
                "notruth",
            ]

            for field in particle_fields:
                x[f"particle_{field}"] = x[f"particle_{field}"][x["particle_valid"]]

            x["particle_valid"] = x["particle_valid"][x["particle_valid"]]

            if len(x["particle_valid"]) < self.cluster_min_num_particles:
                return None

            if len(x["particle_valid"]) > self.cluster_max_num_particles:
                return None

            # Now we have applied the particle cuts, make any cluster cuts
            # that depend on the particles

            # Put scalars into arrays
            x["cluster_id"] = np.array([cluster_id])
            x["cluster_bec"] = np.array([x["cluster_bec"]])
            x["cluster_layer"] = np.array([x["cluster_layer"]])
            x["cluster_size_x"] = np.array([x["cluster_size_x"]])
            x["cluster_size_y"] = np.array([x["cluster_size_y"]])

            x["cluster_valid"] = np.array([True])
            x["cell_valid"] = x["cell_charge"] > 0

        return x


def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value) -> Tensor:
    """Takes a list of tensors, pads them to a given size, and then concatenates them along the a new dimension at zero."""
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)


class PixelClusterCollator:
    def __init__(self, dataset_inputs, dataset_targets, max_num_obj):
        self.dataset_inputs = dataset_inputs
        self.dataset_targets = dataset_targets
        self.max_num_obj = max_num_obj

    def __call__(self, batch):
        inputs, targets = zip(*batch, strict=False)

        hit_max_sizes = {}
        for input_name in self.dataset_inputs:
            hit_max_sizes[input_name] = max(event[f"{input_name}_valid"].shape[-1] for event in inputs)

        batched_inputs = {}
        batched_targets = {}
        for input_name, fields in self.dataset_inputs.items():
            k = f"{input_name}_valid"
            batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), False)

            # Some tasks might require to know hit padding info for loss masking
            batched_targets[k] = batched_inputs[k]

            for field in fields:
                k = f"{input_name}_{field}"

                # Handle scalar case
                if next(iter(inputs))[k].ndim == 1:
                    torch.cat([i[k] for i in inputs], dim=-1)
                    continue

                # If the field is a vector we need to adjust the target size accordingly
                # TODO: Make this nicer
                if next(iter(inputs))[k].ndim == 3:
                    size = (hit_max_sizes[input_name], next(iter(inputs))[k].shape[-1])
                else:
                    size = (hit_max_sizes[input_name],)

                batched_inputs[k] = pad_and_concat([i[k] for i in inputs], size, 0.0)

        for target_name, fields in self.dataset_targets.items():
            if len(target_name.split("_")) == 1:
                size = (self.max_num_obj,)
            else:
                hit = target_name.split("_")[1]
                size = (self.max_num_obj, hit_max_sizes[hit])

            k = f"{target_name}_valid"
            batched_targets[k] = pad_and_concat([t[k] for t in targets], size, False)

            for field in fields:
                k = f"{target_name}_{field}"
                batched_targets[k] = pad_and_concat([t[k] for t in targets], size, 0)

        # Batch the metadata
        batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)

        return batched_inputs, batched_targets


class PixelClusterDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        **kwargs,
    ):
        """Lightning data module. Will iterate over the given directories
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
        # Create training and validation datasets
        if stage == "fit":
            self.train_dset = PixelClusterDataset(dirpath=self.train_dir, num_clusters=self.num_train, **self.kwargs)
            self.val_dset = PixelClusterDataset(dirpath=self.val_dir, num_clusters=self.num_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit":
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = PixelClusterDataset(dirpath=self.test_dir, num_clusters=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: PixelClusterDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=PixelClusterCollator(dataset.inputs, dataset.targets, dataset.cluster_max_num_particles),
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=False)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)
