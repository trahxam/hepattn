# Dataset and DataModule definitions.
# The Dataset handles loading individual events from disk.
# The DataModule handles train/val/test splits and dataloaders.

from torch.utils.data import Dataset

from hepattn.wrappers.data import DataModuleWrapper


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


class MyDataModule(DataModuleWrapper):
    def __init__(self, train_dir, val_dir, num_workers, num_train, num_val, num_test, test_dir=None, batch_size=None, **kwargs):
        super().__init__(
            train_dir=train_dir,
            val_dir=val_dir,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            test_dir=test_dir,
            batch_size=batch_size,
        )
        self.kwargs = kwargs

    def make_dataset(self, dirpath, num_events, split):
        return MyDataset(dirpath=dirpath, num_events=num_events, **self.kwargs)
