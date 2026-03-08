import inspect
from typing import Literal

import torch
from lightning import LightningModule
from lion_pytorch import Lion
from torch import Tensor, nn
from torch._functorch import config as functorch_config  # noqa: PLC2701
from torch.optim import AdamW
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad

from hepattn.utils.types import DictTensor, DoubleNestedDictTensor


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: Literal["AdamW", "Lion"] = "AdamW",
        mtl: bool = False,
        # freeze_except: list[str] | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # self.strict_loading = False

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        self.mtl = mtl

        # if freeze_except is not None:
        #     for name_, param in self.named_parameters():
        #         if not any(pattern in name_ for pattern in freeze_except):
        #             param.requires_grad = False

        if mtl:
            # Donated buffers can cause issues with graph retention needed for MTL
            functorch_config.donated_buffer = False
            # If we are doing multi-task-learning, optimisation step must be done manually
            self.automatic_optimization = False
            # MTL does not currently support intermediate losses
            assert all(task.has_intermediate_loss is False for task in self.model.tasks)

    def forward(self, inputs: DictTensor) -> DoubleNestedDictTensor:
        self._propagate_global_step()
        return self.model(inputs)

    def _propagate_global_step(self) -> None:
        if not hasattr(self.model, "tasks"):
            return
        step = getattr(self, "global_step", None)
        if step is None:
            return
        step_val = int(step)
        for task in self.model.tasks:
            task.global_step = step_val

    def predict(self, outputs: DoubleNestedDictTensor) -> DoubleNestedDictTensor:
        return self.model.predict(outputs)

    def aggregate_losses(self, losses: DoubleNestedDictTensor, stage: str | None = None) -> Tensor:
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
                    self.log(f"{stage}/{layer_name}_{task_name}_{loss_name}", loss_value, sync_dist=True)

            # Log the total loss from the layer
            self.log(f"{stage}/{layer_name}_loss", layer_loss, sync_dist=True)

        # Log the total loss
        self.log(f"{stage}/loss", total_loss, sync_dist=True)
        return total_loss

    def log_task_metrics(self, preds: DoubleNestedDictTensor, targets: DictTensor, stage: str) -> None:
        # Log any task specific metrics
        for task in self.model.tasks:
            # Check that the task actually has some metrics to log
            if not hasattr(task, "metrics"):
                continue

            # Skip tasks that were not active (e.g. during warmup)
            if task.name not in preds["final"]:
                continue

            # Just log the predictions from the final layer for now
            task_metrics = task.metrics(preds["final"][task.name], targets)

            # If the task returned a non-empty metrics dict, log it
            if task_metrics:
                self.log_dict({f"{stage}/final_{task.name}_{k}": v for k, v in task_metrics.items()}, sync_dist=True)

    def log_metrics(self, inputs: DictTensor, preds: DoubleNestedDictTensor, targets: DictTensor, stage: str) -> None:
        # First log any task metrics
        self.log_task_metrics(preds, targets, stage)

        # Log any custom metrics implemented by subclass
        log_custom_metrics = getattr(self, "log_custom_metrics", None)
        if log_custom_metrics is None:
            return

        num_params = len(inspect.signature(log_custom_metrics).parameters)
        if num_params == 4:
            log_custom_metrics(inputs, preds, targets, stage)
        elif num_params == 3:
            log_custom_metrics(preds, targets, stage)
        else:
            raise TypeError("log_custom_metrics must accept either (preds, targets, stage) or (inputs, preds, targets, stage).")

    def _update_task_loss_scales(self) -> None:
        for task in self.model.tasks:
            if hasattr(task, "update_loss_scale"):
                task.update_loss_scale(self.global_step)

    def _update_task_loss_scales(self) -> None:
        for task in self.model.tasks:
            if hasattr(task, "update_loss_scale"):
                task.update_loss_scale(self.global_step)

    def training_step(self, batch: tuple[DictTensor, DictTensor], batch_idx: int) -> DoubleNestedDictTensor | None:
        inputs, targets = batch

        self._update_task_loss_scales()

        # Get the model outputs
        self._propagate_global_step()
        outputs = self.model(inputs)

        # Compute and log losses
        losses, targets = self.model.loss(outputs, targets)

        # Get the predictions from the model, avoid calling predict if possible
        if batch_idx % self.trainer.log_every_n_steps == 0:
            preds = self.predict(outputs)
            self.log_metrics(inputs, preds, targets, "train")

        if self.mtl:
            self.mlt_opt(losses, outputs)
            return None

        total_loss = self.aggregate_losses(losses, stage="train")

        return {"loss": total_loss} | outputs

    def validation_step(self, batch: tuple[DictTensor, DictTensor]) -> DoubleNestedDictTensor:
        inputs, targets = batch

        self._update_task_loss_scales()

        # Get the raw model outputs
        self._propagate_global_step()
        outputs = self.model(inputs)

        # Compute losses then aggregate and log them
        losses, targets = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(inputs, preds, targets, "val")

        return {"loss": total_loss} | outputs

    def test_step(self, batch: tuple[DictTensor, DictTensor]) -> tuple[DoubleNestedDictTensor, ...]:
        inputs, targets = batch
        self._propagate_global_step()
        outputs = self.model(inputs)

        # Calculate loss to also run matching
        losses, targets = self.model.loss(outputs, targets)

        # Get the predictions from the model
        preds = self.model.predict(outputs)

        return outputs, preds, losses

    def on_train_start(self) -> None:
        # Manually overwride the learning rate in case we are starting
        # from a checkpoint that had a LRS and now we want a flat LR
        if self.lrs_config.get("skip_scheduler"):
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lrs_config["initial"]

    def configure_optimizers(self):
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.opt_config['opt']}")

        opt = optimizer(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            # Configure the learning rate scheduler
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            sch = {"scheduler": sch, "interval": "step"}
            return [opt], [sch]

        print("Skipping learning rate scheduler.")
        return opt

    def mlt_opt(self, losses: DictTensor, outputs: DictTensor) -> None:
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
