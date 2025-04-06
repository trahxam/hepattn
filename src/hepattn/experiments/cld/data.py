import numpy as np
from pathlib import Path
import torch
import awkward as ak
import time
from torch.utils.data import DataLoader, Dataset


class CLDDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        event_max_num_particles: int = 1000,
        random_seed: int = 42,
    ):
        super().__init__()

        self.dirpath = dirpath
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.event_max_num_particles = event_max_num_particles
        self.random_seed = random_seed

        # Global random state initialisation
        np.random.seed(random_seed)

        # Setup the number of events that will be used
        event_filenames = list(Path(self.dirpath).glob("*reco*.parquet"))
        num_available_events = len(event_filenames)
        num_requested_events = num_available_events if num_events == -1 else num_events
        self.num_events = min(num_available_events, num_requested_events)

        print(f"Found {num_available_events} available events, {num_requested_events} requested, {self.num_events} used")

        # Allow us to select events by index
        self.event_filenames = event_filenames[:num_events]

    def __len__(self):
        return int(self.num_events)
    
    def __getitem__(self, idx):
        event = self.load_event(idx)

        # TODO: Clean this up, will need to be changed to support hit-particle regression

        inputs = {}
        input_sizes = {}
        for input_name, fields in self.inputs.items():
            field_sizes = set()
            for field in fields:
                x = ak.to_numpy(event[f"{input_name}.{field}"])[0]
                assert np.all(~np.isnan(x))
                field_size = len(x)
                inputs[f"{input_name}_{field}"] = torch.from_numpy(x).unsqueeze(0).half()
                field_sizes.add(field_size)
            
            assert len(field_sizes) == 1, f"Found mismatching field sizes for {input_name}"
            input_sizes[input_name] = next(iter(field_sizes))

            inputs[f"{input_name}_valid"] = torch.ones((1, input_sizes[input_name]), dtype=torch.bool)

        targets = {}
        target_sizes = {}
        for target_name, fields in self.targets.items():

            field_sizes = set()
            for field in fields:
                x = ak.to_numpy(event[f"{target_name}.{field}"])[0]
                field_size = len(x)
                assert np.all(~np.isnan(x))
                x_padding = np.full(self.event_max_num_particles - field_size, np.nan)
                x = np.concatenate((x, x_padding), axis=-1)
                targets[f"{target_name}_{field}"] = torch.from_numpy(x).unsqueeze(0).half()
                field_sizes.add(field_size)
            
            assert len(field_sizes) == 1, f"Found mismatching field sizes for {target_name}"
            target_sizes[target_name] = next(iter(field_sizes))

            targets[f"{target_name}_valid"] = torch.zeros((1, self.event_max_num_particles), dtype=torch.bool)
            targets[f"{target_name}_valid"][:,:target_sizes[target_name]] = True

        # Create the masks that link particles to hits
        for input_name in self.inputs.keys():
            num_hits = input_sizes[input_name]
            mask = np.full((num_hits, self.event_max_num_particles), False)
            # Get the mask indices that map from hits to particles
            mask_idxs = ak.to_numpy(event[f"{input_name}_to_particle_idxs"])[0]
            mask[mask_idxs[:,0],mask_idxs[:,1]] = True
            # Have to transpose the mask to get mask for particles to hits
            targets[f"particle_{input_name}_mask"] = torch.from_numpy(mask.T).unsqueeze(0).bool()
        
        return inputs, targets

    def load_event(self, idx):
        event = ak.from_parquet(self.event_filenames[idx])

        def convert_mm_to_m(i, p):
            # Convert a spatial coordinate from mm to m inplace
            for coord in ["x", "y", "z"]:
                event[f"{i}.{p}.{coord}"] = 0.001 * event[f"{i}.{p}.{coord}"]

        def add_cylindrical_coords(i, p):
            # Add standard tracking cylindrical coordinates
            event[f"{i}.{p}.r"] = np.sqrt(event[f"{i}.{p}.x"]**2 + event[f"{i}.{p}.y"]**2)
            event[f"{i}.{p}.s"] = np.sqrt(event[f"{i}.{p}.x"]**2 + event[f"{i}.{p}.y"]**2 + event[f"{i}.{p}.z"]**2)
            event[f"{i}.{p}.theta"] = np.arccos(event[f"{i}.{p}.z"] / event[f"{i}.{p}.s"])
            event[f"{i}.{p}.eta"] = np.arctanh(event[f"{i}.{p}.z"] / event[f"{i}.{p}.s"])
            event[f"{i}.{p}.phi"] = np.arctan2(event[f"{i}.{p}.y"], event[f"{i}.{p}.x"])
        
        def add_conformal_coords(i, p):
            # Conformal tracking coordinates
            # https://indico.cern.ch/event/658267/papers/2813728/files/8362-Leogrande.pdf
            event[f"{i}.{p}.u"] = event[f"{i}.{p}.x"] / (event[f"{i}.{p}.x"]**2 + event[f"{i}.{p}.y"]**2)
            event[f"{i}.{p}.v"] = event[f"{i}.{p}.y"] / (event[f"{i}.{p}.x"]**2 + event[f"{i}.{p}.y"]**2)

        # Create the input hit objects - only fields that are specified in the config are sent
        for input_name, fields in self.inputs.items():
            add_cylindrical_coords(input_name, "pos")
            add_conformal_coords(input_name, "pos")
            convert_mm_to_m(input_name, "pos")

        # Create the label particle objects
        for target_name, fields in self.targets.items():
            for point in ["vtx", "end"]:
                add_cylindrical_coords(target_name, f"{point}.pos")
                add_conformal_coords(target_name, f"{point}.pos")
                convert_mm_to_m(target_name, f"{point}.pos")
                add_cylindrical_coords(target_name, f"{point}.mom")

        return event