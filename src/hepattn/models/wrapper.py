import torch
from lightning import LightningModule
from torch import nn
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        self.mtl = mtl

        # If we are doing multi-task-learning, optimisation step must be done manually
        if mtl:
            self.automatic_optimization = False

    def forward(self, inputs):
        return self.model(inputs)

    def predict(self, outputs):
        return self.model.predict(outputs)

    def log_losses(self, losses, stage):
        total_loss = 0

        # Log the losses from each task from each layer
        for layer_name, layer_losses in losses.items():
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    self.log(f"{stage}/{layer_name}_{task_name}_{loss_name}", loss_value)
                total_loss += loss_value

        # Log the total loss
        self.log(f"{stage}/loss", total_loss)
        return total_loss

    def log_task_metrics(self, preds, targets, stage):
        # Log any task specific metrics
        for task in self.model.tasks:
            # Check that the task actually has some metrics to log
            if not hasattr(task, "metrics"):
                continue

            # Just log the predictions from the final layer for now
            task_metrics = task.metrics(preds["final"][task.name], targets)

            # If the task returned a non-empty metrics dict, log it
            if task_metrics:
                self.log_dict({f"{stage}/final_{task.name}_{k}": v for k, v in task_metrics.items()})

    def log_metrics(self, preds, targets, stage):
        # First log any task metrics
        self.log_task_metrics(preds, targets, stage)

        # Log any custom metrics implemented by subclass
        if hasattr(self, "log_custom_metrics"):
            self.log_custom_metrics(preds, targets, stage)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # Get the model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        losses = self.model.loss(outputs, targets)
        total_loss = self.log_losses(losses, "train")

        # Get the predictions from the model
        if batch_idx % self.trainer.log_every_n_steps == 0:  # avoid calling predict if possible
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")

        # Use Jacobian Descent for Multi Task Learning https://arxiv.org/abs/2406.16232
        if self.mtl:
            self.mlt_opt(losses, outputs)
            return None

        return total_loss

    def validation_step(self, batch):
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        losses = self.model.loss(outputs, targets)
        total_loss = self.log_losses(losses, "val")

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")

        return total_loss

    def test_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss to also run matching
        losses = self.model.loss(outputs, targets)

        # Get the predictions from the model
        preds = self.model.predict(outputs)

        return outputs, preds, losses

    def on_train_start(self):
        # Manually overwride the learning rate in case we are starting
        # from a checkpoint that had a LRS and now we want a flat LR
        if self.lrs_config["skip_scheduler"]:
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lrs_config["initial"]

    def configure_optimizers(self):
        # Pick which optimizer we want to use
        if self.optimizer.lower() == "adamw":
            from torch.optim import AdamW  # noqa: PLC0415

            opt = AdamW(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])
        # https://arxiv.org/abs/2302.06675
        elif self.optimizer.lower() == "lion":
            from lion_pytorch import Lion  # noqa: PLC0415

            opt = Lion(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config["skip_scheduler"]:
            # Configure the learning rate scheduler
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches * 2,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            sch = {"scheduler": sch, "interval": "step"}
            return [opt], [sch]
        print("Skipping learning rate scheduler.")
        return opt

    def mlt_opt(self, losses, outputs):
        opt = self.optimizers()
        opt.zero_grad()

        for layer_name, layer_losses in losses.items():
            # Get a list of the features that are used by all of the tasks
            layer_feature_names = set()
            for task in self.model.tasks:
                for input_feature in task.inputs:
                    layer_feature_names.add(input_feature)

            # Remove any duplicate features that are used by multiple tasks
            layer_features = [outputs[layer_name][feature_name] for feature_name in layer_feature_names]

            # Perform the backward pass for this layer
            # For each layer we sum the losses from each task, so we get one loss per task
            layer_losses = [sum(losses[layer_name][task.name].values()) for task in self.model.tasks]

            mtl_backward(losses=layer_losses, features=layer_features, aggregator=UPGrad())

        opt.step()
