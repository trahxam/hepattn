
import numpy as np
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from lightning import LightningDataModule, seed_everything
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from hepattn.utils.tensor_utils import pad_to_size

from typing import Dict, List, Any


class LRSMDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        num_samples: int,
        inputs: Dict[str, Dict[str, List[str]]],
        targets: Dict[str, Dict[str, List[str]]],
        input_dtype: str = "float32",
        target_dtype: str = "float32",
        input_pad_value: float = 0.0,
        target_pad_value: float = 0.0,
        force_pad_sizes: Dict[str, int] | None = None,
        skip_pad_items: List[str] | None = None,
        sampling_seed: int = 42,
        sample_reject_warn_limit: int = 10,
    ):
        """
        A PyTorch Dataset that does lazy rejection sampling with memoisation.
        Samples are read lazily, and evaluated on a selection criteria. The result of the
        selection criteria is memoised for speedup after the first epoch.

        Hndles dynamic sample selection, lazy evaluation, padding, and batching for
        data involving variable-sized inputs.

        Parameters
        ----------
        dirpath : str
            Path to the directory containing the dataset samples.
        num_samples : int
            Number of samples to draw from the dataset.
        inputs : dict[dict[str, list]]
            Dictionary mapping input names to lists of feature names.
        targets : dict[dict[str, list]]
            Dictionary mapping target names to lists of target feature names.
        input_dtype : str
            Floating point type used for the inputs
        target_dtype : str
            Floating point type used for the targets
        input_pad_value : float, optional
            Padding value used for missing input values (default is 0.0).
        target_pad_value : float, optional
            Padding value used for missing target values (default is 0.0).
        force_pad_sizes : dict[str, int], optional
            Specifies a size to which objects / hits will be padded to.
            If not specified, an object is padded to ist largest size in the batch.
        skip_pad_items : list[str]
            Specifies a list of objects which should not be padded, 
            e.g. per-sample / event quantites.
        sampling_seed : int, optional
            Random seed used for sampling and reproducibility (default is 42).
        sample_reject_warn_limit : int, optional
            Number of failed sampling attempts before emitting a warning (default is 10).
        """
        if force_pad_sizes is None:
            force_pad_sizes = {}

        if skip_pad_items is None:
            skip_pad_items = []

        super().__init__()

        self.sampling_seed = sampling_seed
        self.dirpath = Path(dirpath)
        self.num_samples = num_samples
        self.inputs = inputs
        self.targets = targets
        self.force_pad_sizes = force_pad_sizes
        self.skip_pad_items = skip_pad_items
        self.input_pad_value = input_pad_value
        self.target_pad_value = target_pad_value
        self.sampling_seed = sampling_seed
        self.sample_reject_warn_limit = sample_reject_warn_limit
        self.dirpath = Path(self.dirpath)

        # Setup input and target datatypes
        dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }

        self.input_dtype = dtypes[input_dtype]
        self.target_dtype = dtypes[target_dtype]

        # Setup random number generators and seeds
        seed_everything(sampling_seed, workers=True)
        self.rng = np.random.default_rng()

        # Map the sample ID to the sample index in the dataset
        # Sample IDs in the values of this must have passed the selection / been accepted
        self.sample_idx_to_sample_id = {}
        # Sample IDs that are known to fail the selection
        self.rejected_sample_ids = []
        # Sample IDs that have not yet been evaluated on the selection
        self.unevaluated_sample_ids = []

        # At the start, we load enough files so that we will have enough for the requested sample size
        # As we read through the dataset, some IDs will be rejected, and so we will load more
        # We will load more samples on demand as needed

    @abstractmethod
    def load_sample(self, sample_id: int) -> dict[str: ndarray] | None:
        """
        Attempts to load a single sample from disk and validate it against selection criteria.

        Parameters
        ----------
        sample_id : int
            Unique identifier of the sample to load.

        Returns
        -------
        dict[str, np.ndarray] or None
            Dictionary of input and target arrays if sample is
            valid and passes the selection, otherwise None.
        """
        pass

    def register_new_samples(self) -> None:
        """
        Loads additional samples into the evaluation pool.

        This method should be overrided so that it populates `self.unevaluated_sample_ids` by
        scanning for new samples not yet marked as accepted or rejected.
        """
        pass

    def __len__(self) -> int:
        """
        Returns the number of samples the dataset should contain.

        Returns
        -------
        int
            Total number of samples.
        """
        return int(self.num_samples)

    def __getitem__(self, sample_idx: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Retrieves a processed sample consisting of input and target tensors for the given index.

        Samples are lazily drawn and filtered to meet selection criteria, padded appropriately,
        and converted to PyTorch tensors of the specified precision.

        Parameters
        ----------
        sample_idx : int
            Index of the sample to retrieve.

        Returns
        -------
        inputs : dict[str, torch.Tensor]
            Dictionary containing input feature tensors with shape (1, ..., feature_dim).
        targets : dict[str, torch.Tensor]
            Dictionary containing target tensors and metadata including 'sample_id'.
        """
        # Attempt to load the sample with the given ID
        # First check if an sample has already been evaluated, accepted, and assigned to this index
        if sample_idx in self.sample_idx_to_sample_id:
            # If its been assigned to an index, we know it passes the selection, and so we can go ahead and load it
            sample_id = self.sample_idx_to_sample_id[sample_idx]
            sample = self.load_sample(sample_id)
        else:
            sample = None

            # Keep trying sample IDs from the set of unevaluated sample IDs until we eventually get an sample that
            # passes the selection
            num_attempts = 1
            while sample is None:
                # Check we still have some samples left
                if len(self.unevaluated_sample_ids) == 0:
                    # If not, we try load in more samples
                    self.register_new_samples()

                    # Check if we now have more samples available
                    # If not we must have ran out of data
                    if len(self.unevaluated_sample_ids) == 0:
                        raise StopIteration("""Ran out of samples that pass the selection,
                            even after attempting to register more samples.""")

                # Randomly sample an ID from the set of unevaluated IDs
                sample_id = self.rng.choice(self.unevaluated_sample_ids)

                # Load this sample ID and see if it passes the selection
                sample = self.load_sample(sample_id)

                # This sample ID has been evaluated on the selection now
                self.unevaluated_sample_ids.remove(sample_id)
                num_attempts += 1

                if num_attempts >= self.sample_reject_warn_limit:
                    print(f"Took {num_attempts} to load sample with index {sample_idx}. Consider a looser selection.")

                # If the sample passes the selection, add it to the accepted list and map it to a dataset index
                if sample is not None:
                    self.sample_idx_to_sample_id[sample_idx] = sample_id
                # If the sample fails the selection, add it to the rejected list and try again
                else:
                    self.rejected_sample_ids.append(sample_id)

        for item_name, fields in (self.inputs | self.targets).items():
            k = f"{item_name}_valid"
            sample[k] = torch.from_numpy(sample[k])

            for field in fields:
                k = f"{item_name}_{field}"
                sample[k] = torch.from_numpy(sample[k])   

        # Apply any manual padding now, need to do it before collate stage incase no batching / collation is used
        for item_name, fields in (self.inputs | self.targets).items():
            # If this item doesnt have any features that are being force padded, continue
            if not any(k in self.force_pad_sizes for k in item_name.split("_")):
                continue

            target_size = tuple(self.force_pad_sizes[k] if k in self.force_pad_sizes else -1 for k in item_name.split("_"))

            # Pad the valid mask, remember that we have a leading singleton batch dim
            k = f"{item_name}_valid"
            sample[k] = pad_to_size(sample[k], target_size, False)

            for field in fields:
                k = f"{item_name}_{field}"

                # Handle the case where the field is a vector by adjusting the target shape accordingly
                if sample[k].dim() - 1 > len(target_size):
                    target_size = (*target_size, sample[k].shape[-1])

                # Check whether this is an input or target so we know the pad value
                pad_value = self.input_pad_value if item_name in self.inputs else self.target_pad_value

                # Now apply the padding                
                sample[k] = pad_to_size(sample[k], target_size, pad_value)

        # Convert to a torch tensor of the correct dtype and add the batch dimension
        inputs = {}
        targets = {}
        for input_name, fields in self.inputs.items():
            inputs[f"{input_name}_valid"] = sample[f"{input_name}_valid"].bool().unsqueeze(0)
            # Some tasks might require to know hit padding info for loss masking
            targets[f"{input_name}_valid"] = inputs[f"{input_name}_valid"]
            for field in fields:
                inputs[f"{input_name}_{field}"] = sample[f"{input_name}_{field}"].to(self.input_dtype).unsqueeze(0)

        # Convert the targets
        for target_name, fields in self.targets.items():
            targets[f"{target_name}_valid"] = sample[f"{target_name}_valid"].bool().unsqueeze(0)
            for field in fields:
                targets[f"{target_name}_{field}"] = sample[f"{target_name}_{field}"].to(self.target_dtype).unsqueeze(0)

        # Convert the metedata
        targets["sample_id"] = torch.tensor(sample_id, dtype=torch.int64)

        return inputs, targets

    def collate_fn(self, batch: list[tuple[dict[str, Tensor], dict[str, Tensor]]]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Collates a list of samples into a single batched sample.

        Handles variable-length inputs by dynamically padding each input and target field
        to the maximum size found in the batch.

        Parameters
        ----------
        batch : list of tuples
            List where each element is a tuple (inputs, targets), as returned by `__getitem__`.

        Returns
        -------
        batched_inputs : dict[str, torch.Tensor]
            Batched and padded input tensors.
        batched_targets : dict[str, torch.Tensor]
            Batched and padded target tensors including 'sample_id'.
        """
        # First unpack the batch, merge the inputs and targets back into one dict
        samples = [inputs | targets for inputs, targets in batch]

        # We first go through all of the items and determine the max size of each item within the batch
        pad_sizes = {}
        for item_name in self.inputs | self.targets:
            # Skip features that have multiple dimensions
            if len(item_name.split("_")) > 1:
                continue

            # Skip features marked as ignored for padding
            if item_name in self.skip_pad_items:
                continue
            
            # Skip features that were manually padded
            if item_name in self.force_pad_sizes:
                continue

            # If no explicit pad size is given for this item, set the pad size to be the max size
            pad_sizes[item_name] = max(sample[f"{item_name}_valid"].shape[1] for sample in samples)

        # Now we apply the padding and the concat
        batched_inputs = {}
        batched_targets = {}
        for item_name, fields in (self.inputs | self.targets).items():
            for sample in samples:
                # If the item was specified as not needing padding, we assume we can just go ahead and concat
                if item_name in self.skip_pad_items:
                    continue

                # The target padding size, ignoring the batch dimension and any field dims
                target_size = tuple(pad_sizes[k] if k in pad_sizes else -1 for k in item_name.split("_"))

                # Pad the valid mask, remember that we have a leading singleton batch dim
                k = f"{item_name}_valid"
                sample[k] = pad_to_size(sample[k], (1, *target_size), False)

                for field in fields:
                    k = f"{item_name}_{field}"

                    # Handle the case where the field is a vector by adjusting the target shape accordingly
                    if sample[k].dim() - 1 > len(target_size):
                        target_size = (*target_size, sample[k].shape[-1])

                    # Check whether this is an input or target so we know the pad value
                    pad_value = self.input_pad_value if item_name in self.inputs else self.target_pad_value

                    # Now apply the padding
                    sample[k] = pad_to_size(sample[k], (1, *target_size), pad_value)

            # Now we can concatenate the samples into one tensor
            k = f"{item_name}_valid"
            batched_inputs[k] = torch.cat([sample[k] for sample in samples], dim=0)
            batched_targets[k] = batched_inputs[k]

            for field in fields:
                k = f"{item_name}_{field}"
                padded = torch.cat([sample[k] for sample in samples], dim=0)

                if item_name in self.inputs:
                    if field in self.inputs[item_name]:
                        batched_inputs[k] = padded

                if item_name in self.targets:
                    if field in self.targets[item_name]:
                        batched_targets[k] = padded

        # Now batch the sample IDs
        batched_targets["sample_id"] = torch.stack([sample["sample_id"] for sample in samples], dim=0)

        return batched_inputs, batched_targets


class LRSMDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_class: LRSMDataset,
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
        """Lightning data module which wraps together test/train/val LRSM datasets.

        Parameters
        ----------
        dataset_class : LRSMDataset
            
        batch_size : int
            Number of samples to read in a minibatch.
        train_dir : str
            Training data directory.
        val_dir : str
            Validation data directory.
        test_dir : str
            Test data directory.
        num_workers : int
            Number of workers / threads too use to read batches.
        num_train : int
            Target number of training samples to load.
        num_val : int
            Target number of validation samples to load.
        num_test : int
            Target number of test samples to load.
        """
        super().__init__()

        self.dataset_class = dataset_class
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
            self.train_dataset = self.dataset_class(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs)
            self.val_dataset = self.dataset_class(dirpath=self.val_dir, num_samples=self.num_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit":
            print(f"Created training dataset with {len(self.train_dataset)} samples")
            print(f"Created validation dataset with {len(self.val_dataset)} samples")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dataset = self.dataset_class(dirpath=self.test_dir, num_samples=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dataset)} samples")

    def get_dataloader(self, stage: str, dataset: LRSMDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, stage="fit", shuffle=False)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=False)
