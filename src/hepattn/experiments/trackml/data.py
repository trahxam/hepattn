import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from pathlib import Path

import hepattn.experiments.trackml.cluster_features as cluster_features


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0

def trackml_event_files_valid(truth_path):
    return all([is_valid_file(str(truth_path).replace("truth", x)) for x in ["truth", "hits", "particles", "cells"]])


class TrackMLDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_samples: int = -1,
        hit_volume_ids: list | None = None,
        particle_min_pt: float = 1.0,
        particle_max_abs_eta: float = 2.5,
        particle_min_num_hits = 3,
        event_max_num_particles = 1000,
        ):
        super().__init__()

        # Set the global random sampling seed
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)  # noqa: NPY002

        # Get a list of truth file paths that have valid accompanying hit/cell/etc info
        files = [file for file in Path(dirpath).glob("event*-truth.csv.gz") if trackml_event_files_valid(file)]

        # Calculate the number of samples that will actually be used
        num_samples_available = len(files)

        if num_samples > num_samples_available:
            msg = f"Requested {num_samples} samples, but only {num_samples_available} are available in the directory {dirpath}."
            raise ValueError(msg)

        if num_samples < 0:
            num_samples = num_samples_available
        
        # Metadata
        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.num_samples = num_samples
        self.files = files[:num_samples]

        self.detector_config_path = self.dirpath.parent / "detectors.csv"
        assert is_valid_file(self.detector_config_path), f"Missing detector config at {self.detector_config_path}"

        # Hit level cuts
        self.hit_volume_ids = hit_volume_ids

        # Particle level cuts
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_min_num_hits = particle_min_num_hits

        # Event level cuts
        self.event_max_num_particles = event_max_num_particles

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        inputs = {}
        targets = {}
        
        # Load the event
        hits, particles = self.load_event(idx)

        num_particles = len(particles)

        # Build the input hits
        for feature, fields in self.inputs.items():
            inputs[f"{feature}_valid"] = torch.full((len(hits),), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]

            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field].values).unsqueeze(0).half()

        # Build the targets for whether a particle slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:len(particles)] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)

        # Build the particle regression targets
        particle_ids = torch.from_numpy(particles["particle_id"].values)
        
        message = f"Event {idx} has {num_particles}, but limit is {self.event_max_num_particles}"
        assert len(particle_ids) <= self.event_max_num_particles, message

        # Fill in empty slots with -1s and get the IDs of the particle on each hit
        particle_ids = torch.cat([particle_ids, -999 * torch.ones(self.event_max_num_particles - len(particle_ids))])
        hit_particle_ids = torch.from_numpy(hits["particle_id"].values)

        # Create the mask targets
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)

        # Create the hit filter targets
        targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"].to_numpy()).unsqueeze(0)

        # Build the regression targets
        for feature, fields in self.targets.items():
            for field in fields:
                x = torch.full((self.event_max_num_particles,), torch.nan)
                x[:num_particles] = torch.from_numpy(particles[field].to_numpy()[: self.event_max_num_particles])
                targets[f"{feature}_{field}"] = x.unsqueeze(0)

        return inputs, targets
    
    def load_event(self, idx):
        truth_fname = self.files[idx]
        hits_fname = Path(str(truth_fname).replace("-truth", "-hits"))
        cells_fname = Path(str(truth_fname).replace("-truth", "-cells"))
        particles_fname = Path(str(truth_fname).replace("-truth", "-particles"))

        # Load in event data
        truth = pd.read_csv(truth_fname, engine="pyarrow")[["hit_id", "particle_id", "weight"]]
        hits = pd.read_csv(hits_fname, engine="pyarrow", dtype={"x": np.float32, "y": np.float32, "z": np.float32},)
        particles = pd.read_csv(particles_fname, engine="pyarrow")
        cells = pd.read_csv(cells_fname, engine="pyarrow")

        assert (truth.index == hits.index).all()

        # Add hit info
        hits["particle_id"] = truth["particle_id"]  # used for evaluation, don't modify

        # Only include hits from the specified volumes
        # pix barrel: 8, pix endcap: 7, 9 - https://competitions.codalab.org/competitions/20112
        if self.hit_volume_ids:
            hits = hits[hits["volume_id"].isin(self.hit_volume_ids)]
        
        # Remove any hits that were cut from the truth and cells also
        truth = truth[truth["hit_id"].isin(hits["hit_id"])]
        cells = cells[cells["hit_id"].isin(hits["hit_id"])]

        # Add additional input information about cluster shapes
        # TODO: Detector config should just be loaded once on init instead of every getitem
        hits = cluster_features.append_cell_features(hits, cells, self.detector_config_path)

        # Scale the input coordinates to in meters so they are ~ 1
        for coord in ["x", "y", "z"]:
            hits[coord] *= 0.01

        # Add extra hit fields
        hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
        hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
        hits["theta"] = np.arccos(hits["z"] / hits["s"])
        hits["phi"] = np.arctan2(hits["y"], hits["x"])
        hits["eta"] = -np.log(np.tan(hits["theta"] / 2))
        hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)

        # Add extra particle fields
        particles["p"] = np.sqrt(particles["px"]**2 + particles["py"]**2 + particles["pz"]**2)
        particles["pt"] = np.sqrt(particles["px"]**2 + particles["py"]**2)
        particles["qopt"] = particles["q"] / particles["pt"]
        particles["eta"] = np.arctanh(particles["pz"] / particles["p"])
        particles["theta"] = np.arccos(particles["pz"] / particles["p"])
        particles["phi"] = np.arctan2(particles["py"], particles["px"])
        particles["costheta"] = np.cos(particles["theta"])
        particles["sintheta"] = np.sin(particles["theta"])
        particles["cosphi"] = np.cos(particles["phi"])
        particles["sinphi"] = np.sin(particles["phi"])

        # Apply particle level cuts based on particle fields
        particles = particles[particles["pt"] > self.particle_min_pt]
        particles = particles[particles["eta"].abs() < self.particle_max_abs_eta]

        # Apply particle cut based on hit content
        counts = hits["particle_id"].value_counts()
        keep_particle_ids = counts[counts >= self.particle_min_num_hits].index.to_numpy()
        particles = particles[particles["particle_id"].isin(keep_particle_ids)]

        # Marck which hits are on a valid / reconstructable particle, for the hit filter
        hits["on_valid_particle"] = hits["particle_id"].isin(particles["particle_id"])

        hits = hits[hits["on_valid_particle"]]

        # Sort hits by phi, needed for phi-windowed attention
        hits = hits.sort_values("phi")

        # Sanity checks
        assert len(particles) != 0, "No particles remaining - loosen selection!"
        assert len(hits) != 0, "No hits remaining - loosen selection!"
        assert particles["particle_id"].nunique() == len(particles), "Non-unique particle ids"

        return hits, particles


class TrackMLDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        pin_memory: bool = True,
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
        **kwargs,
    ):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage == "fit" or stage == "test":
            self.train_dset = TrackMLDataset(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs,)

        if stage == "fit":
            self.val_dset = TrackMLDataset(dirpath=self.val_dir, num_samples=self.num_val, **self.kwargs,)

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = TrackMLDataset(dirpath=self.test_dir, num_samples=self.num_test, trainer=self.trainer, **self.kwargs,)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: TrackMLDataset, shuffle: bool):  # noqa: ARG002
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)
