"""CLD track fitter model wrapper.

BoostedTrackFitter.loss() returns (outputs, targets, losses) — a 3-tuple —
while the base ModelWrapper expects (losses, targets).  Override the step
methods here so the rest of the codebase stays untouched.

Also adds support for task.baseline_metrics() logging (Pandora baseline).
"""

from torch import Tensor, nn

from hepattn.models.wrapper import ModelWrapper


class CLDTrackFitter(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
        pretrained_ckpt_path: str | None = None,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl, pretrained_ckpt_path)

    def log_task_metrics(
        self,
        preds: dict,
        targets: dict[str, Tensor],
        stage: str,
    ) -> None:
        # Per-stage model metrics (stage_0 = helix init, stage_1..n = boost stages)
        for stage_name, stage_preds in preds.items():
            for task in self.model.tasks:
                if not hasattr(task, "metrics"):
                    continue
                if task.name not in stage_preds:
                    continue
                task_metrics = task.metrics(stage_preds[task.name], targets)
                if task_metrics:
                    self.log_dict(
                        {f"{stage}/{stage_name}_{task.name}_{k}": v for k, v in task_metrics.items()},
                        sync_dist=True,
                    )

        # Pandora baseline metrics (once per step, logged under "final")
        for task in self.model.tasks:
            if not hasattr(task, "baseline_metrics"):
                continue
            bm = task.baseline_metrics(targets)
            if bm:
                self.log_dict(
                    {f"{stage}/final_{task.name}_{k}": v for k, v in bm.items()},
                    sync_dist=True,
                )

    def training_step(self, batch: tuple, batch_idx: int):
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)

        if batch_idx % self.trainer.log_every_n_steps == 0:
            preds = self.predict(outputs)
            self.log_task_metrics(preds, targets, "train")

        total_loss = self.aggregate_losses(losses, stage="train")
        return {"loss": total_loss}

    def validation_step(self, batch: tuple):
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")

        preds = self.model.predict(outputs)
        self.log_task_metrics(preds, targets, "val")

        # Cache for TrackResidualPlotCallback
        self._step_preds   = preds
        self._step_targets = targets

        return {"loss": total_loss}

    def test_step(self, batch: tuple):
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        preds = self.model.predict(outputs)
        return outputs, preds, losses
