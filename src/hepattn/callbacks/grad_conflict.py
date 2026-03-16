import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor

HIT_KEYS = ("vtxd", "trkr", "ecal", "hcal", "muon")
CLASS_KEYS = {"object_class_ce"}



def _cosine_sim(a: Tensor, b: Tensor) -> float:
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()


class GradConflictLogger(Callback):
    """Logs cosine similarity between per-hit-group gradients on encoder parameters.

    Does one forward+backward pass per group every `log_every_n_steps` steps.
    Using separate passes (rather than retain_graph=True) avoids incompatibility
    with torch.compile donated buffers.
    """

    def __init__(self, log_every_n_steps: int = 500):
        self.log_every_n_steps = log_every_n_steps
        self._batch: tuple | None = None

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        if trainer.global_step % self.log_every_n_steps == 0:
            self._batch = batch

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        if trainer.global_step % self.log_every_n_steps != 0 or self._batch is None:
            return

        model = pl_module.model
        if not hasattr(model, "encoder") or model.encoder is None:
            return

        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        if not enc_params:
            return

        inputs, targets = self._batch
        self._batch = None

        device = next(model.parameters()).device
        group_names = list(HIT_KEYS) + ["class"]
        group_grads: dict[str, Tensor] = {}

        for group_name in group_names:
            # Fresh forward pass per group — avoids retain_graph entirely
            with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                fwd_outputs = model(inputs)
                losses_nested, _ = model.loss(fwd_outputs, targets)

            group_loss_parts = []
            for layer_losses in losses_nested.values():
                for task_losses in layer_losses.values():
                    for k, v in task_losses.items():
                        if group_name in HIT_KEYS and k.startswith(group_name):
                            group_loss_parts.append(v)
                        elif group_name == "class" and k in CLASS_KEYS:
                            group_loss_parts.append(v)

            if not group_loss_parts:
                continue

            for p in enc_params:
                p.grad = None
            sum(group_loss_parts).backward()

            grads = [p.grad.detach().flatten() for p in enc_params if p.grad is not None]
            if grads:
                group_grads[group_name] = torch.cat(grads)

        # Log all pairwise cosine similarities
        names = sorted(group_grads)
        for i, g1 in enumerate(names):
            for g2 in names[i + 1:]:
                sim = _cosine_sim(group_grads[g1], group_grads[g2])
                pl_module.log(f"train/grad_cos_{g1}_vs_{g2}", sim, on_step=True, on_epoch=False)
