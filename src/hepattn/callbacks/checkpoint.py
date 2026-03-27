from pathlib import Path

import matplotlib.pyplot as plt
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    def __init__(self, monitor: str = "val/loss", log_model: bool = True, **kwargs) -> None:
        filename = "epoch={epoch:03d}-" + monitor.replace("/", "_") + "={" + monitor + ":.5f}"
        super().__init__(save_top_k=-1, monitor=monitor, filename=filename, auto_insert_metric_name=False, **kwargs)
        self.log_model = log_model

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
        self.save_last = False
        self.name = pl_module.name
        if stage == "fit":
            if trainer.fast_dev_run:
                return
            log_dir = Path(trainer.log_dir)
            self.dirpath = str(log_dir / "ckpts")

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        # Save checkpoint inside a named subfolder (e.g. ckpts/epoch=001-.../epoch=001-....ckpt)
        p = Path(filepath)
        ckpt_folder = p.parent / p.stem
        ckpt_folder.mkdir(parents=True, exist_ok=True)
        new_filepath = str(ckpt_folder / p.name)

        super()._save_checkpoint(trainer, new_filepath)

        # Save any event-display figures stored on the module
        pl_module = trainer.lightning_module
        all_event_figs = getattr(pl_module, "_val_display_figs", None)
        if all_event_figs:
            for i, event_figs in enumerate(all_event_figs):
                event_dir = ckpt_folder / "plots" / f"event_{i}"
                event_dir.mkdir(parents=True, exist_ok=True)
                for name, fig in event_figs.items():
                    try:
                        fig.savefig(event_dir / f"event_display_{name}.png", dpi=150, bbox_inches="tight")
                    except Exception as e:
                        print(f"[Checkpoint] Could not save event display '{name}' for event {i}: {e}")
                    finally:
                        plt.close(fig)
            pl_module._val_display_figs = None

        metadata = {"epoch": trainer.current_epoch, "dirpath": str(ckpt_folder)}
        if self.log_model and trainer.logger:
            trainer.logger.experiment.log_model(name=self.name, file_or_folder=new_filepath, metadata=metadata)
