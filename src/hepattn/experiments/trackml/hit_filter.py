import pathlib

import comet_ml  # noqa: F401
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.cli import ArgsType
from torch import nn

from hepattn.experiments.cli import CLI
from hepattn.experiments.trackml.trackml import TrackMLDataModule

config_dir = pathlib.Path(__file__).parent / "configs"


class HitFilter(L.LightningModule):
    def __init__(
        self,
        name: str,
        init: nn.Module,
        encoder: nn.Module,
        dense: nn.Module,
        pos_enc: nn.Module | None = None,
        target: str = "hit_tgt",
        lrs_config: dict | None = None,
        torch_compile: bool = False,
    ):
        super().__init__()

        self.name = name
        self.init = init
        self.encoder = encoder
        self.dense = dense
        self.target = target
        self.lrs_config = lrs_config
        self.times: list[float] = []
        self.num_hits: list[int] = []
        self.pos_enc = pos_enc
        if torch_compile:
            self.init = torch.compile(init, dynamic=True)
            self.encoder = torch.compile(encoder, dynamic=True)
            self.dense = torch.compile(dense, dynamic=True)

    def on_train_start(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.log_hyperparams({"trainable_params": params})

    def forward(self, x, labels=None, timing=False):
        if timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        x = self.init(x["hit"])

        if self.pos_enc:
            pe = self.pos_enc(labels["phi"].unsqueeze(-1))
            x += pe

        x = self.encoder(x)
        preds = self.dense(x).squeeze(-1)

        if timing:
            end.record()
            torch.cuda.synchronize()
            self.times.append(start.elapsed_time(end))
            self.num_hits.append(x.shape[1])

        loss = None
        if labels:
            tgt = labels["hit"][self.target]
            weight = 1 / tgt.float().mean()
            loss = F.binary_cross_entropy_with_logits(preds, tgt.type_as(preds), pos_weight=weight)

        return {"hit_pred": preds}, {"hit_pred": loss}

    def shared_step(self, batch, **kwargs):
        inputs, labels = batch
        preds, loss = self(inputs, labels, **kwargs)
        loss["loss"] = sum(subloss for subloss in loss.values())
        return preds, labels, loss

    def training_step(self, batch):
        preds, labels, loss = self.shared_step(batch)
        self.log_losses(loss, stage="train")
        self.log_metrics(preds, labels, stage="train")
        return loss["loss"]

    def validation_step(self, batch):
        preds, labels, loss = self.shared_step(batch)
        self.log_losses(loss, stage="validate")
        self.log_metrics(preds, labels, stage="validate")
        return loss

    def test_step(self, batch):
        preds, _, _ = self.shared_step(batch, timing=True)
        return preds

    def log_losses(self, loss, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}
        self.log(f"{stage}/loss", loss["loss"], prog_bar=True, **kwargs)

    def log_metrics(self, preds, labels, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}

        pred = preds["hit_pred"]
        tgt = labels["hit"]["tgt_pid"].bool()
        pred_true = pred.sigmoid() > 0.15
        self.log(f"{stage}/nh_total_pre", float(pred.shape[1]), **kwargs)
        self.log(f"{stage}/nh_total_post", float(pred_true.sum()), **kwargs)
        self.log(f"{stage}/nh_pred_true", pred_true.float().sum(), **kwargs)
        self.log(f"{stage}/nh_pred_false", (~pred_true).float().sum(), **kwargs)
        self.log(f"{stage}/nh_valid_pre", tgt.float().sum(), **kwargs)
        self.log(f"{stage}/nh_valid_post", (pred_true & tgt).float().sum(), **kwargs)
        self.log(f"{stage}/nh_noise_pre", (~tgt).float().sum(), **kwargs)
        self.log(f"{stage}/nh_noise_post", (pred_true & ~tgt).float().sum(), **kwargs)

        # accuracy
        self.log(f"{stage}/acc", (pred_true == tgt).half().mean(), **kwargs)

        # precision and recall
        tp = (pred_true * tgt).sum()
        self.log(f"{stage}/valid_recall", tp / tgt.sum(), **kwargs)
        self.log(f"{stage}/valid_precision", tp / pred_true.sum(), **kwargs)

        tn = ((~pred_true) * (~tgt)).sum()
        self.log(f"{stage}/noise_recall", tn / (~tgt).sum(), **kwargs)
        self.log(f"{stage}/noise_precision", tn / (~pred_true).sum(), **kwargs)

    def on_test_epoch_end(self):
        if self.times:
            self.times = self.times[5:]
            self.num_hits = self.num_hits[5:]
            mean_time = sum(self.times) / len(self.times)
            std_time = torch.tensor(self.times).std()
            print(f"Average inference time: {mean_time:.2f} ms ± {std_time:.2f} ms")
            print(len(self.times), len(self.num_hits))
            np.save(f"times/{self.name}_times.npy", self.times)
            np.save(f"times/{self.name}_num_hits.npy", self.num_hits)
            print(f"Saved timing info to times/{self.name}_times.npy")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lrs_config["initial"], weight_decay=1e-5)

        # 1cycle
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


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=HitFilter,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
