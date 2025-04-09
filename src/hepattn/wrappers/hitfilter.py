import torch
import torch.nn.functional as F
from lightning import LightningModule
from lion_pytorch import Lion
from torch import nn


class HitFilterWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        init_net: nn.Module,
        encoder: nn.Module,
        dense: nn.Module,
        target_name: str,
        opt_config: dict | None = None,
    ):
        super().__init__()
        # TODO: None of this will work for multiple input types

        self.name = name
        self.init_net = init_net
        self.encoder = encoder
        self.dense = dense
        self.input_name = self.init_net.input_name
        self.target_name = target_name
        self.opt_config = opt_config

        # Used for logging inference times
        self.times: list[float] = []
        self.num_hits: list[int] = []

    def forward(self, inputs: dict) -> dict:
        x_embed = self.init_net(inputs)
        x_embed = self.encoder(x_embed)
        logits = self.dense(x_embed).squeeze(-1)
        outputs = {"final": {"hit_filter": {f"{self.input_name}_logits": logits}}}
        return outputs
    
    def predict(self, outputs: dict) -> dict:
        pred = outputs["final"]["hit_filter"][f"{self.input_name}_logits"] >= 0.1
        preds = {"final": {"hit_filter": {f"{self.input_name}_valid": pred}}}
        return preds

    def loss(self, outputs: dict, targets: dict) -> dict:
        target = targets[self.target_name]
        output = outputs["final"]["hit_filter"][f"{self.input_name}_logits"]
        weight = 1 / target.float().mean()

        loss = F.binary_cross_entropy_with_logits(output, target.type_as(output), pos_weight=weight)
        losses = {"final": {"hit_filter": {f"{self.input_name}_bce": loss}}}
        return losses

    def step(self, batch, **kwargs):
        inputs, targets = batch
        outputs = self.forward(inputs, targets, **kwargs)
        losses = self.loss(outputs, targets)
        return outputs, targets, losses

    def training_step(self, batch):
        outputs, targets, losses = self.step(batch)
        self.log_losses(losses, stage="train")
        self.log_metrics(preds, targets, stage="train")
        return loss["loss"]

    def validation_step(self, batch):
        preds, targets, loss = self.step(batch)
        self.log_losses(loss, stage="validate")
        self.log_metrics(preds, targets, stage="validate")
        return loss

    def test_step(self, batch):
        preds, targets, loss = self.step(batch, timing=True)
        return preds

    def log_losses(self, loss, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}
        self.log(f"{stage}/loss", loss["loss"], prog_bar=True, **kwargs)

    def log_metrics(self, preds, targets, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}

        pred = preds["hit_pred"]
        tgt = targets["hit"]["tgt_pid"].bool()
        pred_true = pred.sigmoid() > 0.1

        self.log(f"{stage}/nh_total_pre", float(pred.shape[1]), **kwargs)
        self.log(f"{stage}/nh_total_post", float(pred_true.sum()), **kwargs)
        self.log(f"{stage}/nh_pred_true", pred_true.float().sum(), **kwargs)
        self.log(f"{stage}/nh_pred_false", (~pred_true).float().sum(), **kwargs)
        self.log(f"{stage}/nh_valid_pre", tgt.float().sum(), **kwargs)
        self.log(f"{stage}/nh_valid_post", (pred_true & tgt).float().sum(), **kwargs)
        self.log(f"{stage}/nh_noise_pre", (~tgt).float().sum(), **kwargs)
        self.log(f"{stage}/nh_noise_post", (pred_true & ~tgt).float().sum(), **kwargs)

        # Accuracy
        self.log(f"{stage}/acc", (pred_true == tgt).half().mean(), **kwargs)

        # Precision and recall
        tp = (pred_true * tgt).sum()
        self.log(f"{stage}/valid_recall", tp / tgt.sum(), **kwargs)
        self.log(f"{stage}/valid_precision", tp / pred_true.sum(), **kwargs)

        tn = ((~pred_true) * (~tgt)).sum()
        self.log(f"{stage}/noise_recall", tn / (~tgt).sum(), **kwargs)
        self.log(f"{stage}/noise_precision", tn / (~pred_true).sum(), **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam

        if self.opt_config["opt"] == "AdamW":
            optimizer = torch.optim.AdamW
        elif self.opt_config["opt"] == "Lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.opt_config['opt']}")
        
        opt = optimizer(self.parameters(), lr=self.opt_config["initial_lr"], weight_decay=self.opt_config["weight_decay"])

        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.opt_config["max_lr"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.opt_config["max_lr"] / self.opt_config["initial_lr"],
            final_div_factor=self.opt_config["initial_lr"] / self.opt_config["final_lr"],
            pct_start=float(self.opt_config["pct_start"]),
        )
        sch = {"scheduler": sch, "interval": "step"}

        return [opt], [sch]
