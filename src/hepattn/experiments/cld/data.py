import numpy as np
from pathlib import Path
import torch
import time
from torch.utils.data import DataLoader, Dataset


class CLDDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        labels: dict,
        num_events: int = -1,
        num_particles: int = 250,
        random_seed: int = 42,
    ):
        super().__init__()

        self.dirpath = dirpath
        self.inputs = inputs
        self.labels = labels
        self.num_events = num_events
        self.num_particles = num_particles
        self.random_seed = random_seed

        # Global random state initialisation
        np.random.seed(random_seed)

        # Setup the number of events that will be used
        event_filenames = list(Path(self.dirpath).glob("*event*.npy"))
        num_available_events = len(event_filenames)
        num_requested_events = num_available_events if num_events == -1 else num_events
        self.num_events = min(num_available_events, num_requested_events)

        print(f"Found {num_available_events} available events, {num_requested_events} requested, {self.num_events} used")

        # Allow us to select events by index
        self.event_filenames = event_filenames[:num_events]

    def __len__(self):
        return int(self.num_events)
    
    def __getitem__(self, idx):
        event = np.load(self.event_filenames[idx], allow_pickle=True)[()]

        def convert_mm_to_m(i, p):
            # Convert a spatial coordinate from mm to m inplace
            for coord in ["x", "y", "z"]:
                event[f"{i}_{p}.{coord}"] = 0.001 * event[f"{i}_{p}.{coord}"]

        def add_cylindrical_coords(i, p):
            # Add standard tracking cylindrical coordinates
            event[f"{i}_{p}.r"] = np.sqrt(event[f"{i}_{p}.x"]**2 + event[f"{i}_{p}.y"]**2)
            event[f"{i}_{p}.s"] = np.sqrt(event[f"{i}_{p}.x"]**2 + event[f"{i}_{p}.y"]**2 + event[f"{i}_{p}.z"]**2)
            event[f"{i}_{p}.theta"] = np.arccos(event[f"{i}_{p}.z"] / event[f"{i}_{p}.s"])
            event[f"{i}_{p}.eta"] = np.arctanh(event[f"{i}_{p}.z"] / event[f"{i}_{p}.s"])
            event[f"{i}_{p}.phi"] = np.arctan2(event[f"{i}_{p}.y"], event[f"{i}_{p}.x"])
        
        def add_conformal_coords(i, p):
            # Conformal tracking coordinates
            # https://indico.cern.ch/event/658267/papers/2813728/files/8362-Leogrande.pdf
            event[f"{i}_{p}.u"] = event[f"{i}_{p}.x"] / (event[f"{i}_{p}.x"]**2 + event[f"{i}_{p}.y"]**2)
            event[f"{i}_{p}.v"] = event[f"{i}_{p}.y"] / (event[f"{i}_{p}.x"]**2 + event[f"{i}_{p}.y"]**2)

        # Create the input hit objects - only fields that are specified in the config are sent
        inputs = {}
        for input_name, fields in self.inputs.items():
            add_cylindrical_coords(input_name, "position")
            add_conformal_coords(input_name, "position")
            convert_mm_to_m(input_name, "position")

            for field in fields:
                inputs[f"{input_name}_{field}"] = torch.from_numpy(event[f"{input_name}_{field}"])

        # Create the label particle objects
        labels = {}
        for label_name, fields in self.labels.items():
            add_cylindrical_coords(label_name, "vertex")
            add_conformal_coords(label_name, "vertex")
            convert_mm_to_m(label_name, "vertex")

            add_cylindrical_coords(label_name, "endpoint")
            add_conformal_coords(label_name, "endpoint")
            convert_mm_to_m(label_name, "endpoint")

            add_cylindrical_coords(label_name, "momentum")
            add_cylindrical_coords(label_name, "momentumAtEndpoint")

            for field in fields:
                labels[f"{label_name}_{field}"] = torch.from_numpy(event[f"{label_name}_{field}"])

        # Create the masks that link particles to hits
        masks = {}
        for input_name in self.inputs.keys():
            mask = np.full((event[f"{input_name}_position.x"].shape[0], self.num_particles), False)
            # Get the mask indices that map from hits to particles
            mask_idxs = event[f"{input_name}_to_particle_idxs"]
            mask[mask_idxs[:,0],mask_idxs[:,1]] = True
            # Have to transpose the mask to get mask for particles to hits
            masks[f"particle_{input_name}"] = mask.T
        
        

        return inputs, labels, masks




import yaml

config = yaml.safe_load(Path("config.yaml").read_text())

dataset = CLDDataset(
    dirpath="test_data_out",
    inputs=config["data"]["inputs"],
    labels=config["data"]["labels"],
    num_events=50,
    num_particles=250,
)


dataset[0]