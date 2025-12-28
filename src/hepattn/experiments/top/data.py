import uproot
import awkward as ak
import numpy as np
import torch

from typing import Any
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule


JET_FIELDS = ["pt", "eta", "phi", "e", "m", "bTag", "truthmatch"]
JET_BRANCHES = [f"jet_{f}" for f in JET_FIELDS]

PARTICLE_FIELDS = ["pid", "pt", "eta", "phi", "m"]
PARTICLE_BRANCHES = [f"particle_{f}" for f in PARTICLE_FIELDS]


class TopEventDataset(Dataset):
    def __init__(
        self,
        path: str,
        tree="Delphes;1",
        num_events: int = -1,
        event_min_num_b_jets: int = 2,
        ):

        path = Path(path)

        with uproot.open(path) as f:
            arr = f[tree].arrays(library="ak")

        event_mask = ak.sum(arr["jet_bTag"], axis=-1) >= event_min_num_b_jets
        arr = arr[event_mask]

        num_available = len(arr)
        num_requested = num_events if num_events >= 0 else num_available
        self.num_events = int(min(num_available, num_requested))
        self.data = arr[: self.num_events]

        print(f"Found {num_available} available events, using {self.num_events}")

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __getitems__(self, indices):
        return self.data[indices]

def collate_pad_events(batch, pad_value=0.0):
    events = batch if isinstance(batch, ak.Array) else ak.Array(batch)

    # ---- Jets (pad to max_jets in batch) ----
    jets = events[JET_BRANCHES]

    jet_lengths = torch.as_tensor(ak.to_numpy(ak.num(jets["jet_pt"], axis=1)), dtype=torch.long)
    max_jets = int(jet_lengths.max().item()) if len(jet_lengths) else 0

    jets["jet_pt_tev"] = jets["jet_pt"] / 1000.0
    jets["jet_m_tev"] = jets["jet_m"] / 1000.0
    jets["jet_e_tev"] = jets["jet_e"] / 1000.0

    eps = 1e-3

    def floor_with_mask(x, eps):
        m = x > eps
        mf = ak.values_astype(m, np.float32)
        return x * mf + eps * (1.0 - mf)

    pt_safe = floor_with_mask(jets["jet_pt"], eps)
    m_safe  = floor_with_mask(jets["jet_m"],  eps)
    e_safe  = floor_with_mask(jets["jet_e"],  eps)

    jets = ak.with_field(jets, np.log(pt_safe), "jet_log_pt")
    jets = ak.with_field(jets, np.log(m_safe),  "jet_log_m")
    jets = ak.with_field(jets, np.log(e_safe),  "jet_log_e")

    jets["jet_px"] = jets["jet_pt"] * np.cos(jets["jet_phi"])
    jets["jet_py"] = jets["jet_pt"] * np.sin(jets["jet_phi"])
    jets["jet_pz"] = jets["jet_pt"] * np.sinh(jets["jet_eta"])

    jets_pad = ak.fill_none(ak.pad_none(jets, max_jets, axis=1), pad_value)

    jet_valid = (
        (torch.arange(max_jets)[None, :] < jet_lengths[:, None])
        if max_jets
        else torch.zeros((len(jet_lengths), 0), dtype=torch.bool)
    )

    inputs = {
        "event_num_jets": jet_lengths,
        "jet_valid": jet_valid,
    }

    # per-jet event-level counts
    event_num_btag = torch.as_tensor(
        ak.to_numpy(ak.sum(jets["jet_bTag"], axis=1)), dtype=torch.long
    )

    jet_event_num_jets = jet_lengths[:, None].expand(-1, max_jets).to(torch.float32)
    jet_event_num_btag = event_num_btag[:, None].expand(-1, max_jets).to(torch.float32)

    # zero out padded jet slots
    jet_event_num_jets = jet_event_num_jets.masked_fill(~jet_valid, 0.0)
    jet_event_num_btag = jet_event_num_btag.masked_fill(~jet_valid, 0.0)


    inputs["jet_event_num_jets"] = jet_event_num_jets
    inputs["jet_event_num_btag"] = jet_event_num_btag

    # convert jet fields
    for field in jets_pad.fields:
        if field == "jet_truthmatch":
            continue
        inputs[field] = torch.from_numpy(ak.to_numpy(jets_pad[field])).float()

    # ---- Pairwise jet-jet features (B, J, J) ----
    eta = inputs["jet_eta"]
    phi = inputs["jet_phi"]

    deta = eta[:, :, None] - eta[:, None, :]
    dphi_raw = phi[:, :, None] - phi[:, None, :]
    dphi = torch.atan2(torch.sin(dphi_raw), torch.cos(dphi_raw))
    dr = torch.sqrt(deta * deta + dphi * dphi)

    E  = inputs["jet_e"]
    px = inputs["jet_px"]
    py = inputs["jet_py"]
    pz = inputs["jet_pz"]

    Eij  = E[:, :, None]  + E[:, None, :]
    pxij = px[:, :, None] + px[:, None, :]
    pyij = py[:, :, None] + py[:, None, :]
    pzij = pz[:, :, None] + pz[:, None, :]

    m2 = Eij * Eij - pxij * pxij - pyij * pyij - pzij * pzij
    mjj = torch.sqrt(torch.clamp(m2, min=0.0))

    pair_valid = jet_valid[:, :, None] & jet_valid[:, None, :]
    #diag = torch.eye(max_jets, dtype=torch.bool).unsqueeze(0)
    #pair_valid = pair_valid & ~diag  # exclude i==j

    inputs["jet_jet_valid"] = pair_valid
    inputs["jet_jet_mjj"] = mjj.masked_fill(~pair_valid, float(pad_value))
    inputs["jet_jet_log_mjj"] = torch.log(torch.clip(inputs["jet_jet_mjj"], 1e-3, None))
    inputs["jet_jet_deta"] = deta.masked_fill(~pair_valid, float(pad_value))
    inputs["jet_jet_dphi"] = dphi.masked_fill(~pair_valid, float(pad_value))
    inputs["jet_jet_dr"] = dr.masked_fill(~pair_valid, float(pad_value))

        # targets from truthmatch: (B, 6, J) where row i corresponds to label (i+1)
    tm = torch.as_tensor(ak.to_numpy(jets_pad["jet_truthmatch"]), dtype=torch.long)  # (B, J)

    labels = torch.arange(1, 7, dtype=tm.dtype, device=tm.device).view(1, 6, 1)   # (1, 6, 1)
    jet_truth_mask = (tm[:, None, :] == labels)                                    # (B, 6, J)

    # ensure padded jets are always false
    jet_truth_mask = jet_truth_mask & inputs["jet_valid"][:, None, :]

    targets = {
        "part_jet_valid": jet_truth_mask,      # (B, 6, J)
        "jet_valid": inputs["jet_valid"],
        "top_valid": torch.full((inputs["jet_valid"].shape[0], 2), True, dtype=torch.bool),
    }

    # (optional) keep your old 2xJ top masks derived from the 6-way mask
    top1 = jet_truth_mask[:, 0:3, :].any(dim=1)   # labels 1,2,3
    top2 = jet_truth_mask[:, 3:6, :].any(dim=1)   # labels 4,5,6
    targets["top_jet_valid"] = torch.stack([top1, top2], dim=1)  # (B, 2, J)


    targets["event_num_jets"] = torch.as_tensor(ak.to_numpy(events["njet"]), dtype=torch.float32)
    targets["event_num_btag"] = torch.as_tensor(ak.to_numpy(events["nbTagged"]), dtype=torch.float32)
    targets["event_all_matched"] = torch.as_tensor(ak.to_numpy(events["allMatchedEvent"]), dtype=torch.bool)
    targets["event_least_one_matched_w"] = torch.as_tensor(ak.to_numpy(events["leastOneMatchedW"]), dtype=torch.bool)

    # ---- Particles (pad to max_particles in batch) ----
    parts = events[PARTICLE_BRANCHES]

    par_lengths = torch.as_tensor(ak.to_numpy(ak.num(parts["particle_pt"], axis=1)), dtype=torch.long)
    max_parts = int(par_lengths.max().item()) if len(par_lengths) else 0

    parts_pad = ak.fill_none(ak.pad_none(parts, max_parts, axis=1), pad_value)

    targets["particle_num"] = par_lengths
    targets["particle_valid"] = (
        (torch.arange(max_parts)[None, :] < par_lengths[:, None])
        if max_parts
        else torch.zeros((len(par_lengths), 0), dtype=torch.bool)
    )

    targets["particle_pid"] = torch.from_numpy(ak.to_numpy(parts_pad["particle_pid"])).to(torch.int32)
    targets["particle_pt"]  = torch.from_numpy(ak.to_numpy(parts_pad["particle_pt"])).float()
    targets["particle_eta"] = torch.from_numpy(ak.to_numpy(parts_pad["particle_eta"])).float()
    targets["particle_phi"] = torch.from_numpy(ak.to_numpy(parts_pad["particle_phi"])).float()
    targets["particle_m"]   = torch.from_numpy(ak.to_numpy(parts_pad["particle_m"])).float()

    targets["particle_px"] = targets["particle_pt"] * torch.cos(targets["particle_phi"]).float()
    targets["particle_py"] = targets["particle_pt"] * torch.sin(targets["particle_phi"]).float()

    return inputs, targets


class TopEventDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        num_workers: int,
        num_train: int = -1,
        num_val: int = -1,
        num_test: int = -1,
        batch_size: int = 1,
        test_path: str | None = None,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        drop_last_train: bool = False,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
        # dataset / collate kwargs
        tree: str = "Delphes;1",
        event_min_num_b_jets: int = 2,
        pad_value: float = 0.0,
        **dataloader_kwargs: Any,
    ):
        super().__init__()

        self.train_path = str(train_path)
        self.val_path = str(val_path)
        self.test_path = str(test_path) if test_path is not None else None

        self.num_workers = int(num_workers)
        self.num_train = int(num_train)
        self.num_val = int(num_val)
        self.num_test = int(num_test)
        self.batch_size = int(batch_size)

        self.pin_memory = bool(pin_memory)
        self.shuffle_train = bool(shuffle_train)
        self.drop_last_train = bool(drop_last_train)

        # If not specified: only keep persistent workers when workers > 0
        self.persistent_workers = (
            (self.num_workers > 0) if persistent_workers is None else bool(persistent_workers)
        )
        self.prefetch_factor = prefetch_factor

        self.tree = tree
        self.event_min_num_b_jets = int(event_min_num_b_jets)
        self.pad_value = float(pad_value)

        self.dataloader_kwargs = dict(dataloader_kwargs)

        # will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        stage = stage or "fit"

        if stage in ("fit", "validate"):
            self.train_dataset = TopEventDataset(
                path=self.train_path,
                tree=self.tree,
                num_events=self.num_train,
                event_min_num_b_jets=self.event_min_num_b_jets,
            )
            self.val_dataset = TopEventDataset(
                path=self.val_path,
                tree=self.tree,
                num_events=self.num_val,
                event_min_num_b_jets=self.event_min_num_b_jets,
            )
            print(f"Created training dataset with {len(self.train_dataset):,} events")
            print(f"Created validation dataset with {len(self.val_dataset):,} events")

        if stage in ("test", "predict"):
            assert self.test_path is not None, "No test file specified; pass test_path=..."
            self.test_dataset = TopEventDataset(
                path=self.test_path,
                tree=self.tree,
                num_events=self.num_test,
                event_min_num_b_jets=self.event_min_num_b_jets,
            )
            print(f"Created test dataset with {len(self.test_dataset):,} events")

    def _make_loader(self, dataset, *, shuffle: bool, drop_last: bool):
        kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=lambda batch: collate_pad_events(batch, pad_value=self.pad_value),
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            **self.dataloader_kwargs,
        )

        # prefetch_factor is only valid when num_workers > 0
        if self.prefetch_factor is not None and self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**kwargs)

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=self.shuffle_train, drop_last=self.drop_last_train)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False, drop_last=False)
