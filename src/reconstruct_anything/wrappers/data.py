from __future__ import annotations

from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset


class DataModuleWrapper(LightningDataModule):
    """Base DataModule for reconstruct-anything experiments.

    Handles the standard train/val/test split setup, DataLoader creation,
    and logging. Subclasses implement make_dataset() to provide the
    experiment-specific Dataset.

    Args:
        train_dir: Directory containing training data.
        val_dir: Directory containing validation data.
        num_workers: Number of DataLoader worker processes.
        num_train: Number of training events (-1 for all).
        num_val: Number of validation events (-1 for all).
        num_test: Number of test events (-1 for all).
        test_dir: Directory containing test data.
        pin_memory: Whether to pin memory for faster GPU transfer.
        batch_size: DataLoader batch size. Use None for single-event
            batches (when the Dataset already adds a batch dimension).
    """

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
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.kwargs: dict = {}

    def make_dataset(self, dirpath: str, num_events: int, split: str) -> Dataset:
        """Create and return a Dataset for the given split.

        Args:
            dirpath: Path to data directory for this split.
            num_events: Number of events to load (-1 for all).
            split: One of "train", "val", or "test".

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        """Instantiate datasets for the requested stage.

        Args:
            stage: Lightning stage - "fit", "validate", or "test".
        """
        if stage in {"fit", "test"}:
            self.train_dataset = self.make_dataset(self.train_dir, self.num_train, "train")
        if stage in {"fit", "validate"}:
            self.val_dataset = self.make_dataset(self.val_dir, self.num_val, "val")
        if stage == "test":
            if self.test_dir is None:
                msg = "test_dir must be set for test stage"
                raise ValueError(msg)
            self.test_dataset = self.make_dataset(self.test_dir, self.num_test, "test")

        if stage == "fit":
            rank_zero_info(f"Train dataset: {len(self.train_dataset):,} events")
            rank_zero_info(f"Val dataset:   {len(self.val_dataset):,} events")
        if stage == "test":
            rank_zero_info(f"Test dataset:  {len(self.test_dataset):,} events")

    def get_dataloader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        """Build a DataLoader for the given dataset.

        Args:
            dataset: Dataset to wrap.
            shuffle: Whether to shuffle the data.

        Returns:
            Configured DataLoader.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return self.get_dataloader(self.test_dataset, shuffle=False)
