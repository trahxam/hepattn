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
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        self.mtl = mtl

        if mtl:
            # Donated buffers can cause issues with graph retention needed for MTL
            functorch_config.donated_buffer = False
            # If we are doing multi-task-learning, optimisation step must be done manually
            self.automatic_optimization = False

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
            raise TypeError(
                "log_custom_metrics must accept either (preds, targets, stage) "
                "or (inputs, preds, targets, stage)."
            )

    def training_step(self, batch: tuple[DictTensor, DictTensor], batch_idx: int) -> DoubleNestedDictTensor | None:
        inputs, targets = batch

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

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        clip_val = self.lrs_config.get("gradient_clip_val") or gradient_clip_val
        if clip_val:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

    def mlt_opt(self, losses: DictTensor, outputs: DictTensor) -> None:
        opt = self.optimizers()
        opt.zero_grad()

        # Backprop intermediate layer losses normally — they provide auxiliary supervision
        # for the decoder but don't participate in conflict resolution
        intermediate_losses = [
            loss_value
            for layer_name, layer_losses in losses.items()
            if layer_name != "final"
            for task_losses in layer_losses.values()
            for loss_value in task_losses.values()
        ]
        if intermediate_losses:
            sum(intermediate_losses).backward(retain_graph=True)

        # Split final layer losses into tracking (si-hit + classification) vs calo groups
        tracker_prefixes = ("vtxd", "trkr", "object")
        calo_prefixes = ("ecal", "hcal", "muon")
        tracking_losses, calo_losses = [], []
        for task_losses in losses["final"].values():
            if not isinstance(task_losses, dict):
                continue
            for loss_name, loss_value in task_losses.items():
                if any(loss_name.startswith(p) for p in tracker_prefixes):
                    tracking_losses.append(loss_value)
                elif any(loss_name.startswith(p) for p in calo_prefixes):
                    calo_losses.append(loss_value)

        task_loss_groups = [
            g for g in [
                sum(tracking_losses) if tracking_losses else None,
                sum(calo_losses) if calo_losses else None,
            ]
            if g is not None
        ]

        features = [outputs["final"]["query_embed"], outputs["final"]["key_embed"]]

        if len(task_loss_groups) > 1:
            mtl_backward(losses=task_loss_groups, features=features, aggregator=UPGrad(), retain_graph=True)
        elif len(task_loss_groups) == 1:
            task_loss_groups[0].backward()

        clip_val = self.lrs_config.get("gradient_clip_val") or self.trainer.gradient_clip_val
        if clip_val:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

        opt.step()
