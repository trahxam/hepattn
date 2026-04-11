from pathlib import Path

import h5py
from lightning import Callback, LightningModule, Trainer
from torch import Tensor

from reconstruct_anything.utils.tensor_utils import tensor_to_numpy


class PredictionWriter(Callback):
    """Callback that writes model inputs, outputs, predictions, targets, and losses to an HDF5 file.

    Each sample is stored in a dedicated group keyed by its sample ID.  Items
    produced by the model (outputs, preds, losses) are nested under
    ``layer_name/task_name/field_name`` sub-groups, while inputs and targets are
    stored directly under ``sample_id/inputs`` and ``sample_id/targets``.

    Attributes:
        write_inputs: Whether to write raw model inputs.
        write_outputs: Whether to write raw model outputs (logits, etc.).
        write_preds: Whether to write post-processed predictions.
        write_targets: Whether to write ground-truth targets.
        write_losses: Whether to write per-sample losses.
        write_layers: Names of decoder layers whose outputs to write (default: ``["final"]``).
    """

    def __init__(
        self,
        write_inputs: bool,
        write_outputs: bool,
        write_preds: bool,
        write_targets: bool,
        write_losses: bool,
        write_layers: list[str] | None = None,
    ):
        """Initialise the writer with flags controlling which data are persisted.

        Args:
            write_inputs: Whether to write raw model inputs.
            write_outputs: Whether to write raw model outputs.
            write_preds: Whether to write post-processed predictions.
            write_targets: Whether to write ground-truth targets.
            write_losses: Whether to write per-sample losses.
            write_layers: Decoder layer names to include in outputs/preds/losses.
                Defaults to ``["final"]``.
        """
        if write_layers is None:
            write_layers = ["final"]
        super().__init__()

        self.write_inputs = write_inputs
        self.write_outputs = write_outputs
        self.write_preds = write_preds
        self.write_targets = write_targets
        self.write_losses = write_losses
        self.write_layers = write_layers

        self.file = None
        self.num_queries: int | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Open the output HDF5 file when entering the test stage."""
        if stage != "test":
            return

        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)

        self.trainer = trainer
        self.dataset = trainer.datamodule.test_dataloader().dataset

        self.num_queries = self._resolve_num_queries(pl_module)

        # Open the handle for writing to the file
        self.file = h5py.File(self.output_path, "w")

    def _resolve_num_queries(self, pl_module: LightningModule) -> int:
        """Read the number of decoder queries from the model."""
        # User assumption: model.decoder._num_queries is always available.
        return int(pl_module.model.decoder._num_queries)  # noqa: SLF001

    @property
    def output_path(self) -> Path:
        """Return the path for the output HDF5 file, placed alongside the checkpoint."""
        # The output dataset will be saved in the same directory as the checkpoint
        split = Path(self.dataset.dirpath).name
        return Path(self.trainer.ckpt_dir / f"{self.trainer.ckpt_name}_{split}_eval.h5")

    def on_test_batch_end(self, trainer, pl_module, test_step_outputs, batch, batch_idx):
        """Write each sample in the batch to the output HDF5 file."""
        inputs, targets = batch
        outputs, preds, losses = test_step_outputs

        # handle batched case
        if "sample_id" in targets:
            # Get all of the sample IDs in the batch, this is what will be used to retrieve the samples
            sample_ids = targets["sample_id"]

            # Iterate through all of the samples in the batch
            for idx, sample_id in enumerate(sample_ids):
                self.write_sample(sample_id, inputs, targets, outputs, preds, losses, idx)

        # handle unbatched case
        else:
            self.write_sample(batch_idx, inputs, targets, outputs, preds, losses, 0)

    def write_sample(self, sample_id, inputs, targets, outputs, preds, losses, idx):
        """Write a single sample to the output file."""
        # create a group for thie sample_id
        if isinstance(sample_id, Tensor):
            sample_id = sample_id.item()
        sample_group = self.file.create_group(str(sample_id))

        # Write inputs and targets
        if self.write_inputs:
            self.write_items(sample_group, "inputs", inputs, idx)

        if self.write_targets:
            self.write_items(sample_group, "targets", targets, idx)

        # Items produced by model have layer/task structure
        if self.write_outputs:
            self.write_layer_task_items(sample_group, "outputs", outputs, idx)

        if self.write_preds:
            self.write_layer_task_items(sample_group, "preds", preds, idx)

        if self.write_losses:
            self.write_layer_task_items(sample_group, "losses", losses, idx)

    def write_items(self, sample_group, item_name, items, idx):
        """Write a flat dict of tensors under ``sample_group/item_name``."""
        # This will write out a dict of items that has the structure
        # sample/item/value, e.g.
        # sample_id/inputs/pixel_x
        items_group = sample_group.create_group(item_name)
        for name, value in items.items():
            self.create_dataset(items_group, name, value[idx][None, ...])

    def write_layer_task_items(self, sample_group, item_name, items, idx):
        """Write a nested layer/task/field dict under ``sample_group/item_name``."""
        items_group = sample_group.create_group(item_name)
        # This will write out a dict of items that has the structure
        # sample/item/layer/task/value, e.g.
        # sample_id/preds/final/track_regression/track_phi
        for layer_name, layer_items in items.items():
            # Only write items fow the specified layers
            if layer_name not in self.write_layers:
                continue
            layer_group = items_group.create_group(layer_name)
            for task_name, task_items in layer_items.items():
                task_group = layer_group.create_group(task_name)
                for name, value in task_items.items():
                    self.create_dataset(task_group, name, value[idx][None, ...])

    def create_dataset(self, group, name, value):
        """Convert a tensor to numpy and write it as an LZF-compressed HDF5 dataset."""
        # Shouldn't need to detach as we are testing
        value = tensor_to_numpy(value)

        # Write the data to the file
        group.create_dataset(name, data=value, compression="lzf")

    def teardown(self, trainer, module, stage):
        """Close the HDF5 file handle at the end of the test stage."""
        # Close the file handle now we are done
        if stage == "test":
            if self.file is not None:
                self.file.close()
            print("-" * 80)
            print("Created output file", self.output_path)
            print("-" * 80)
