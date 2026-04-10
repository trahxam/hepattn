from collections.abc import Callable

import torch
from lightning import LightningModule
from lightning.pytorch.cli import OptimizerCallable
from torch import Tensor, nn
from torch._functorch import config as functorch_config  # noqa: PLC2701
from torch.optim import AdamW
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        optimizer: OptimizerCallable = AdamW,
        lr_scheduler: Callable | None = None,
        mtl: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mtl = mtl

        if mtl:
            # Donated buffers can cause issues with graph retention needed for MTL
            functorch_config.donated_buffer = False
            # If we are doing multi-task-learning, optimisation step must be done manually
            self.automatic_optimization = False
            # MTL does not currently support intermediate losses
            assert all(task.has_intermediate_loss is False for task in self.model.tasks)

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model.predict(outputs)

    def aggregate_losses(self, losses: dict[str, dict[str, dict[str, Tensor]]], stage: str | None = None) -> Tensor:
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)

        for layer_name, layer_losses in losses.items():
            layer_loss = 0
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    self.log(f"{stage}/{layer_name}_{task_name}_{loss_name}", loss_value, sync_dist=True)
                    total_loss += loss_value
                    layer_loss += loss_value

            # Log the total loss from the layer
            self.log(f"{stage}/{layer_name}_loss", layer_loss, sync_dist=True)

        # Log the total loss
        self.log(f"{stage}/loss", total_loss, sync_dist=True)
        return total_loss

    def log_task_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        # Log any task specific metrics
        for layer_name in preds:
            # Determine which task list to use based on layer name
            if hasattr(self.model, "decoder"):
                tasks = self.model.decoder.encoder_tasks if layer_name == "encoder" else self.model.decoder.tasks
            else:
                tasks = self.model.tasks
            for task in tasks:
                if task.name not in preds[layer_name]:
                    continue

                task_metrics = task.metrics(preds[layer_name][task.name], targets)
                if task_metrics:
                    self.log_dict({f"{stage}/{layer_name}_{task.name}_{k}": v for k, v in task_metrics.items()}, sync_dist=True)

    def log_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        self.log_task_metrics(preds, targets, stage)

        if hasattr(self, "log_custom_metrics"):
            self.log_custom_metrics(preds, targets, stage)

    def training_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]], batch_idx: int) -> dict[str, Tensor] | None:
        inputs, targets = batch

        # Get the model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        outputs, targets, losses = self.model.loss(outputs, targets)

        # Get the predictions from the model, avoid calling predict if possible
        if batch_idx % self.trainer.log_every_n_steps == 0:
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")

        if self.mtl:
            self.mlt_opt(losses, outputs)
            return None
        total_loss = self.aggregate_losses(losses, stage="train")

        return {"loss": total_loss}

    def validation_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> dict[str, Tensor]:
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Compute losses then aggregate and log them
        outputs, targets, losses = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")

        return {"loss": total_loss}

    def test_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss to also run matching
        outputs, targets, losses = self.model.loss(outputs, targets)

        # Get the predictions from the model
        preds = self.model.predict(outputs)

        return outputs, preds, losses

    def configure_optimizers(self):
        opt = self.optimizer(self.model.parameters())
        if self.lr_scheduler is None:
            return opt
        sch = self.lr_scheduler(opt, total_steps=self.trainer.estimated_stepping_batches)
        return [opt], [{"scheduler": sch, "interval": "step"}]

    def mlt_opt(self, losses: dict[str, Tensor], outputs: dict[str, Tensor]) -> None:
        opt = self.optimizers()
        opt.zero_grad()

        # TODO: Make this not hard coded?
        feature_names = ["query_embed", "key_embed"]

        # Remove any duplicate features that are used by multiple tasks
        features = [outputs["final"][feature_name] for feature_name in feature_names]

        # TODO: Figure out if we can set retain_graph to false somehow, since it uses a lot of memory
        task_losses = [sum(losses["final"][task.name].values()) for task in self.model.tasks]
        mtl_backward(losses=task_losses, features=features, aggregator=UPGrad(), retain_graph=True)

        # Manually perform the optimizer step
        opt.step()
