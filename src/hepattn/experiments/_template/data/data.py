# Dataset and DataModule definitions.
# The Dataset handles loading individual events from disk.
# The DataModule handles train/val/test splits and dataloaders.

from lightning import LightningDataModule
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dirpath, **kwargs):
        super().__init__()
        self.dirpath = dirpath

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # Return (inputs, targets) dicts with tensors keyed like:
        #   inputs:  {"{object}_valid": ..., "{object}_{field}": ...}
        #   targets: {"{object}_valid": ..., "{object}_{field}": ...}
        raise NotImplementedError


class MyDataModule(LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, batch_size=1, num_workers=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError
