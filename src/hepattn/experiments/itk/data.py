import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from pathlib import Path


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class ITkDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_volume_ids: list | None = None,
        particle_min_pt: float = 1.0,
        particle_max_abs_eta: float = 2.5,
        particle_min_num_pixels = 3,
        particle_min_num_strips = 6,
        event_max_num_particles = 1000,
        ):
        super().__init__()

        # Set the global random sampling seed
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)  # noqa: NPY002

        # Get a list of event names
        event_names = [Path(file).stem.replace("-parts", "") for file in Path(dirpath).glob("*-parts.parquet")]

        # Calculate the number of events that will actually be used
        num_events_available = len(event_names)

        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available in the directory {dirpath}."
            raise ValueError(msg)

        if num_events < 0:
            num_events = num_events_available
        
        # Metadata
        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.event_names = event_names[:num_events]

        # Hit level cuts
        self.hit_volume_ids = hit_volume_ids

        # Particle level cuts
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_min_num_pixels = particle_min_num_pixels
        self.particle_min_num_strips = particle_min_num_strips

        # Event level cuts
        self.event_max_num_particles = event_max_num_particles

    def __len__(self):
        return int(self.num_events)

    def __getitem__(self, idx):
        inputs = {}
        targets = {}
        
        # Load the event
        hits, particles = self.load_event(idx)

        num_particles = len(particles)

        # Build the input hits
        for input_name, fields in self.inputs.items():
            inputs[f"{input_name}_valid"] = torch.full((len(hits[input_name]),), True).unsqueeze(0)
            targets[f"{input_name}_valid"] = inputs[f"{input_name}_valid"]

            for field in fields:
                inputs[f"{input_name}_{field}"] = torch.from_numpy(hits[input_name][field].values).unsqueeze(0).half()


        # Build the targets for whether a particle slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:len(particles)] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)

        # Build the particle regression targets
        particle_ids = torch.from_numpy(particles["particle_id"].values).long()
        
        message = f"Event {idx} has {num_particles}, but limit is {self.event_max_num_particles}"
        assert len(particle_ids) <= self.event_max_num_particles, message

        # Fill in empty slots with -1s and get the IDs of the particle on each hit
        # Important! Make sure that the null value tensor we concatenate on the end is also a long tensor,
        # otherwise the resulting tensor will be a 32 bit int tensor which causes overflow issues as 
        # particle_ids can be larger than the int32 max value
        particle_ids = torch.cat([particle_ids, -1 * torch.ones(self.event_max_num_particles - len(particle_ids)).long()])

        torch.set_printoptions(threshold=10000, sci_mode=False)

        for hit in ["pixel", "strip"]:
            hit_particle_ids = torch.from_numpy(hits[hit]["particle_id"].values).long()
            

            # Create the mask targets
            targets[f"particle_{hit}_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)

            # Create the hit filter targets
            targets[f"{hit}_on_valid_particle"] = torch.from_numpy(hits[hit]["on_valid_particle"].to_numpy()).unsqueeze(0)

        # Build the regression targets
        for feature, fields in self.targets.items():
            for field in fields:
                x = torch.full((self.event_max_num_particles,), torch.nan)
                x[:num_particles] = torch.from_numpy(particles[field].to_numpy()[: self.event_max_num_particles])
                targets[f"{feature}_{field}"] = x.unsqueeze(0)

        return inputs, targets
    
    def load_event(self, idx):
        # Load in event data
        event_name = self.event_names[idx]

        particles = pd.read_parquet(self.dirpath / Path(event_name + "-parts.parquet"))
        pixel = pd.read_parquet(self.dirpath / Path(event_name + "-pixel.parquet"))
        strip = pd.read_parquet(self.dirpath / Path(event_name + "-strip.parquet"))

        

        # Only include hits from the specified volumes
        # pix barrel: 8, pix endcap: 7, 9 - https://competitions.codalab.org/competitions/20112
        #if self.hit_volume_ids:
        #    pixel = pixel[pixel["region"].isin(self.hit_volume_ids)]
        #    strip = strip[strip["region"].isin(self.hit_volume_ids)]

        # Add extra hit fields
        for hits in [pixel, strip]:
            # Scale the input coordinates to in meters so they are ~ 1
            for coord in ["x", "y", "z"]:
                hits[coord] *= 0.01

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
        particles["qopt"] = particles["charge"] / particles["pt"]
        particles["eta"] = np.arctanh(particles["pz"] / particles["p"])
        particles["theta"] = np.arccos(particles["pz"] / particles["p"])
        particles["phi"] = np.arctan2(particles["py"], particles["px"])
        particles["costheta"] = np.cos(particles["theta"])
        particles["sintheta"] = np.sin(particles["theta"])
        particles["cosphi"] = np.cos(particles["phi"])
        particles["sinphi"] = np.sin(particles["phi"])

        # Apply particle level cuts based on particle fields
        particles = particles[particles["pt"] >= self.particle_min_pt]
        particles = particles[particles["eta"].abs() <= self.particle_max_abs_eta]

        particles = particles[particles["particle_id"] != 0]

        # Marck which hits are on a valid / reconstructable particle, for the hit filter
        pixel["on_valid_particle"] = pixel["particle_id"].isin(particles["particle_id"])
        strip["on_valid_particle"] = strip["particle_id"].isin(particles["particle_id"])

        pixel = pixel[pixel["on_valid_particle"]]
        strip = strip[strip["on_valid_particle"]]

        # Sanity checks
        assert len(particles) != 0, "No particles remaining - loosen selection!"
        assert len(pixel) != 0, "No pixel hits remaining - loosen selection!"
        assert len(strip) != 0, "No strip hits remaining - loosen selection!"
        assert particles["particle_id"].nunique() == len(particles), "Non-unique particle ids"

        hits = {"pixel": pixel, "strip": strip}

        return hits, particles


class ITkDataModule(LightningDataModule):
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
            self.train_dset = ITkDataset(dirpath=self.train_dir, num_events=self.num_train, **self.kwargs,)

        if stage == "fit":
            self.val_dset = ITkDataset(dirpath=self.val_dir, num_events=self.num_val, **self.kwargs,)

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = ITkDataset(dirpath=self.test_dir, num_events=self.num_test, trainer=self.trainer, **self.kwargs,)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: ITkDataset, shuffle: bool):  # noqa: ARG002
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
