from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer


class HitPredictionWriter(Callback):
    """Legacy callback for writing hit predictions and targets to an HDF5 file.

    Each event is stored as a separate group keyed by its event ID from the
    dataset.  Use ``PredictionWriter`` for new code.
    """

    def __init__(self) -> None:
        """Initialise the writer (no configurable parameters)."""
        super().__init__()

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        """Resolve the test dataset reference when entering the test stage."""
        if stage != "test":
            return

        self.trainer = trainer

        # get test dataset
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.num_events = len(self.ds)

    @property
    def output_path(self) -> Path:
        """Return the output file path, placed alongside the checkpoint."""
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        return Path(out_dir / f"{out_basename}__test.h5")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Open the output HDF5 file at the start of testing."""
        self.file = h5py.File(self.output_path, "w")

    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx):
        """Write hit predictions and targets for the current batch to the HDF5 file."""
        preds = outputs
        targets = batch[1]

        # create a group for each event using the event id
        event_id = self.ds.get_event_id_from_batch_idx(batch_idx)
        g = self.file.create_group(event_id)

        self.create_dataset(g, preds["hit_pred"], "hit_pred")
        for i in targets:
            if isinstance(targets[i], dict):
                for j in targets[i]:
                    self.create_dataset(g, targets[i][j], f"targets/{i}/{j}")
            else:
                self.create_dataset(g, targets[i], "targets/" + i)

    def on_test_epoch_end(self, trainer, module):
        """Close the HDF5 file after all test batches have been written."""
        self.file.close()
        print("Created output file", self.output_path)
        print("-" * 100, "\n")

    def create_dataset(self, f, a, name, half_precision=True):
        """Convert a tensor to (optionally half-precision) numpy and write as a compressed dataset.

        Args:
            f: HDF5 group or file to write into.
            a: Tensor or numpy array to store.
            name: Dataset name within ``f``.
            half_precision: If ``True``, downcast float32 arrays to float16.
        """
        a = a.squeeze()

        if isinstance(a, torch.Tensor):
            a = a.float().cpu().numpy()

        # convert down to float16
        if half_precision:
            t = np.dtype(a.dtype)
            if t.kind == "f" and t.itemsize == 2:
                a = a.astype(np.float16)

        # write
        f.create_dataset(name, data=a, compression="lzf")
