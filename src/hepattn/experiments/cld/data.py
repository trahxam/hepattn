from pathlib import Path

import awkward as ak
import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class CLDDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        merge_inputs: dict[str, list[str]] | None = None,
        num_events: int = -1,
        particle_min_pt: float = 0.1,
        include_neutral: bool = True,
        include_charged: bool = True,
        charged_particle_min_num_hits: dict[str, int] | None = None,
        charged_particle_max_num_hits: dict[str, int] | None = None,
        neutral_particle_min_num_hits: dict[str, int] | None = None,
        neutral_particle_max_num_hits: dict[str, int] | None = None,
        particle_max_hit_deflection_deta: dict[str, float] | None = None,
        particle_max_hit_deflection_dphi: dict[str, float] | None = None,
        truth_filter_hits: list[str] | None = None,
        event_max_num_particles: int = 256,
        random_seed: int = 42,
    ):
        if truth_filter_hits is None:
            truth_filter_hits = []
        if particle_max_hit_deflection_dphi is None:
            particle_max_hit_deflection_dphi = {}
        if particle_max_hit_deflection_deta is None:
            particle_max_hit_deflection_deta = {}
        if neutral_particle_max_num_hits is None:
            neutral_particle_max_num_hits = {}
        if neutral_particle_min_num_hits is None:
            neutral_particle_min_num_hits = {}
        if charged_particle_max_num_hits is None:
            charged_particle_max_num_hits = {}
        if charged_particle_min_num_hits is None:
            charged_particle_min_num_hits = {}
        if merge_inputs is None:
            merge_inputs = {}
        super().__init__()

        self.dirpath = dirpath
        self.inputs = inputs
        self.targets = targets
        self.merge_inputs = merge_inputs
        self.num_events = num_events
        self.particle_min_pt = particle_min_pt
        self.include_neutral = include_neutral
        self.include_charged = include_charged
        self.charged_particle_min_num_hits = charged_particle_min_num_hits
        self.charged_particle_max_num_hits = charged_particle_max_num_hits
        self.neutral_particle_min_num_hits = neutral_particle_min_num_hits
        self.neutral_particle_max_num_hits = neutral_particle_max_num_hits
        self.particle_max_hit_deflection_deta = particle_max_hit_deflection_deta
        self.particle_max_hit_deflection_dphi = particle_max_hit_deflection_dphi
        self.truth_filter_hits = truth_filter_hits
        self.event_max_num_particles = event_max_num_particles
        self.random_seed = random_seed

        # Global random state initialisation
        np.random.default_rng(42)

        # Setup the number of events that will be used
        event_filenames = list(Path(self.dirpath).glob("*reco*.parquet"))
        num_available_events = len(event_filenames)
        num_requested_events = num_available_events if num_events == -1 else num_events
        self.num_events = min(num_available_events, num_requested_events)

        print(f"Found {num_available_events} available events, {num_requested_events} requested, {self.num_events} used")

        # Allow us to select events by index
        self.event_filenames = event_filenames[: self.num_events]

    def __len__(self):
        return int(self.num_events)

    def load_event(self, idx):
        """Loads a single CLD event from a preprocessed parquet file."""

        event_raw = ak.from_parquet(self.event_filenames[idx])

        # First unpack from awkward parquet format to dict of numpy arrays
        event = {}
        for key in event_raw.fields:
            event[key] = ak.to_numpy(event_raw[key])[0]

        def convert_mm_to_m(i, p):
            # Convert a spatial coordinate from mm to m inplace
            for coord in ["x", "y", "z"]:
                event[f"{i}.{p}.{coord}"] *= 0.001

        def add_cylindrical_coords(i, p):
            # Add standard tracking cylindrical coordinates
            event[f"{i}.{p}.r"] = np.sqrt(event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2)
            event[f"{i}.{p}.s"] = np.sqrt(event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2 + event[f"{i}.{p}.z"] ** 2)
            event[f"{i}.{p}.theta"] = np.arccos(event[f"{i}.{p}.z"] / event[f"{i}.{p}.s"])
            event[f"{i}.{p}.eta"] = -np.log(np.tan(event[f"{i}.{p}.theta"] / 2))
            event[f"{i}.{p}.phi"] = np.arctan2(event[f"{i}.{p}.y"], event[f"{i}.{p}.x"])

        def add_conformal_coords(i, p):
            # Conformal tracking coordinates
            event[f"{i}.{p}.u"] = event[f"{i}.{p}.x"] / (event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2)
            event[f"{i}.{p}.v"] = event[f"{i}.{p}.y"] / (event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2)

        hits = ["vtb", "vte", "itb", "ite", "otb", "ote", "ecb", "ece", "hcb", "hce", "hco", "muon"]

        # Add extra coordinates to the hits
        for hit in hits:
            # It is important to do the mm -> m conversion first, so that all other
            # distance fields are also in m, which is required to not to cause nans in the positional encoding
            convert_mm_to_m(hit, "pos")
            add_cylindrical_coords(hit, "pos")
            add_conformal_coords(hit, "pos")

        # Add extra coordinates to the start and end points of particles
        for point in ["vtx", "end"]:
            convert_mm_to_m("particle", f"{point}.pos")
            add_cylindrical_coords("particle", f"{point}.pos")
            add_conformal_coords("particle", f"{point}.pos")
            add_cylindrical_coords("particle", f"{point}.mom")

        # Merge inputs, first check all requested merged inputs have the same
        # fields and that the fields are given in the same order
        if self.merge_inputs:
            for merged_input_name, input_names in self.merge_inputs.items():
                merged_input_fields = self.inputs[merged_input_name]

                # Concatenate the fields from all of the inputs that make the merged input
                for field in merged_input_fields:
                    event[f"{merged_input_name}.{field}"] = np.concatenate([event[f"{input_name}.{field}"] for input_name in input_names], axis=-1)

        # Particle count includes invaliud particles, since the linking indices were built
        # before any of these particle selections were made
        event["particle_valid"] = np.full_like(event["particle.PDG"], True, np.bool)
        num_particles = len(event["particle_valid"])

        # Now we will construct the masks that link particles to hits
        for hit in hits:
            num_hits = len(event[f"{hit}.type"])
            mask = np.full((num_hits, num_particles), False)

            # Has shape (num hit-particle links, 2)
            mask_idxs = event[f"{hit}_to_particle_idxs"]

            # Get the mask indices that map from hits to particles
            # Check there are actually some hits present and there is at least one particle to link
            # Indices link hits to particles, so have to transpose to get particles to hits
            if num_hits > 0 and len(mask_idxs) > 0:
                mask[mask_idxs[:, 0], mask_idxs[:, 1]] = True

            event[f"particle_{hit}_valid"] = mask.T

        # Merge together any masks
        if self.merge_inputs:
            for merged_input_name, input_names in self.merge_inputs.items():
                event[f"particle_{merged_input_name}_valid"] = np.concatenate([event[f"particle_{hit}_valid"] for hit in input_names], axis=-1)

        # Add extra labels for particles
        event["particle.isCharged"] = np.abs(event["particle.charge"]) > 0
        event["particle.isNeutral"] = ~event["particle.isCharged"]

        # Set which particles we deem to be targets / reconstructable
        particle_cuts = {"min_pt": event["particle.vtx.mom.r"] >= self.particle_min_pt}

        if not self.include_charged:
            particle_cuts["not_charged"] = ~event["particle.isCharged"]

        if not self.include_neutral:
            particle_cuts["not_neutral"] = ~event["particle.isNeutral"]

        # Now we have built the masks, we can apply hit/counting based cuts
        for hit_name, min_num_hits in self.charged_particle_min_num_hits.items():
            particle_cuts[f"charged_min_{hit_name}"] = ~(event["particle.isCharged"] & (event[f"particle_{hit_name}_valid"].sum(-1) < min_num_hits))

        for hit_name, max_num_hits in self.charged_particle_max_num_hits.items():
            particle_cuts[f"charged_max_{hit_name}"] = ~(event["particle.isCharged"] & (event[f"particle_{hit_name}_valid"].sum(-1) > max_num_hits))

        for hit_name, min_num_hits in self.neutral_particle_min_num_hits.items():
            particle_cuts[f"neutral_min_{hit_name}"] = ~(event["particle.isNeutral"] & (event[f"particle_{hit_name}_valid"].sum(-1) < min_num_hits))

        for hit_name, max_num_hits in self.neutral_particle_max_num_hits.items():
            particle_cuts[f"neutral_max_{hit_name}"] = ~(event["particle.isNeutral"] & (event[f"particle_{hit_name}_valid"].sum(-1) > max_num_hits))

        # Apply hit deflection based cuts
        # If two hits that are subsequent in time have a difference in eta/phi larger than some maximum,
        # then any particles on this hit are cut
        for hit_name, max_deta in self.particle_max_hit_deflection_deta.items():
            mask = event[f"particle_{hit_name}_valid"]
            eta = np.ma.masked_array(mask * event[f"{hit_name}.pos.eta"][..., None, :], mask=~mask)
            time = np.ma.masked_array(mask * event[f"{hit_name}.time"][..., None, :], mask=~mask)
            idx = np.ma.argsort(time, axis=-1)
            eta_sorted = np.take_along_axis(eta, idx, axis=-1)
            deta = np.ma.diff(eta_sorted, axis=-1)
            particle_cuts["hit_deflection_eta"] = np.ma.all(np.abs(deta) < max_deta, axis=-1)

        # Now also apply the hit deflection cut in phi
        for hit_name, max_dphi in self.particle_max_hit_deflection_dphi.items():
            mask = event[f"particle_{hit_name}_valid"]
            phi = np.ma.masked_array(mask * event[f"{hit_name}.pos.phi"][..., None, :], mask=~mask)
            time = np.ma.masked_array(mask * event[f"{hit_name}.time"][..., None, :], mask=~mask)
            idx = np.ma.argsort(time, axis=-1)
            phi_sorted = np.take_along_axis(phi, idx, axis=-1)
            dphi = np.ma.diff(phi_sorted, axis=-1)
            particle_cuts["hit_deflection_phi"] = np.ma.all(np.abs(dphi) < max_dphi, axis=-1)

        # Apply the particle cuts
        for cut_mask in particle_cuts.values():
            event["particle_valid"] &= cut_mask

        # Remove any mask slots for invalid particles
        for input_name in self.inputs:
            event[f"particle_{input_name}_valid"] &= event["particle_valid"][:, np.newaxis]

        # Do truth hit filtering if specified
        for input_name in self.truth_filter_hits:
            # Get hits that are not noise
            mask = event[f"particle_{input_name}_valid"].any(-2)

            # First drop noise hits from inputs
            for field in self.inputs[input_name]:
                event[f"{input_name}.{field}"] = event[f"{input_name}.{field}"][mask]

            # Also drop noise hits from the target masks
            if f"particle_{input_name}" in self.targets:
                event[f"particle_{input_name}_valid"] = event[f"particle_{input_name}_valid"][:, mask]

        # Check that there are no noise hits if we specified this
        for input_name in self.truth_filter_hits:
            if f"particle_{input_name}" in self.targets:
                assert np.all(event[f"particle_{input_name}_valid"].sum(-2) > 0)

        # Pick out the inputs that have actually been requested
        inputs = {}
        for input_name, fields in self.inputs.items():
            inputs[f"{input_name}_valid"] = ~np.isnan(event[f"{input_name}.type"])
            for field in fields:
                inputs[f"{input_name}_{field}"] = event[f"{input_name}.{field}"]

        # Now pick out the targets
        targets = {}
        for target_name, fields in self.targets.items():
            targets[f"{target_name}_valid"] = event[f"{target_name}_valid"]
            for field in fields:
                targets[f"{target_name}_{field}"] = event[f"{target_name}.{field}"]

        return inputs, targets

    def __getitem__(self, idx):
        inputs, targets = self.load_event(idx)

        # Convert to a torch tensor of the correct dtype and add the batch dimension
        inputs_out = {}
        targets_out = {}
        for input_name, fields in self.inputs.items():
            inputs_out[f"{input_name}_valid"] = torch.from_numpy(inputs[f"{input_name}_valid"]).bool().unsqueeze(0)
            # Some tasks might require to know hit padding info for loss masking
            targets_out[f"{input_name}_valid"] = inputs_out[f"{input_name}_valid"]
            for field in fields:
                inputs_out[f"{input_name}_{field}"] = torch.from_numpy(inputs[f"{input_name}_{field}"]).half().unsqueeze(0)

        for target_name, fields in self.targets.items():
            targets_out[f"{target_name}_valid"] = torch.from_numpy(targets[f"{target_name}_valid"]).bool().unsqueeze(0)
            for field in fields:
                targets_out[f"{target_name}_{field}"] = torch.from_numpy(targets[f"{target_name}_{field}"]).half().unsqueeze(0)

        return inputs_out, targets_out


def pad_to_size(x: torch.Tensor, d: tuple, value) -> torch.Tensor:
    """
    Pads the tensor x on the right side of each dimension to match the size given in d.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        d (tuple): Target size for each dimension (must have same length as x.dim()).

    Returns:
        torch.Tensor: Padded tensor with shape == d.
    """

    if len(d) != x.dim():
        raise ValueError(f"Target size must match input tensor dimensions: {x.shape} vs {d}")

    padding = []
    for i in reversed(range(x.dim())):
        pad_len = d[i] - x.size(i)
        if pad_len < 0:
            raise ValueError(f"Cannot pad dimension {i} from {x.size(i)} to {d[i]} (target smaller than current).")
        padding.extend([0, pad_len])  # (left, right) padding — pad only on the right

    return F.pad(x, padding, value=value)


def pad_and_concat(items, target_size, pad_value):
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)


class CLDCollator:
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
                batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), 0.0)

        for target_name, fields in self.dataset_targets.items():
            if target_name == "particle":
                size = (self.max_num_obj,)
            else:
                hit = target_name.split("_")[1]
                size = (self.max_num_obj, hit_max_sizes[hit])

            k = f"{target_name}_valid"
            batched_targets[k] = pad_and_concat([i[k] for i in targets], size, False)

            for field in fields:
                k = f"{target_name}_{field}"
                batched_targets[k] = pad_and_concat([i[k] for i in targets], size, torch.nan)

        return batched_inputs, batched_targets


class CLDDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        batch_size: int = 1,
        test_dir: str | None = None,
        pin_memory: bool = True,
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
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage == "fit" or stage == "test":
            self.train_dset = CLDDataset(dirpath=self.train_dir, num_events=self.num_train, **self.kwargs)

        if stage == "fit":
            self.val_dset = CLDDataset(dirpath=self.val_dir, num_events=self.num_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit":
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = CLDDataset(dirpath=self.test_dir, num_events=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: CLDDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=CLDCollator(dataset.inputs, dataset.targets, dataset.event_max_num_particles),
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
