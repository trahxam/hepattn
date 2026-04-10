import torch


class OneCycleLR:
    """Configurable OneCycleLR scheduler where total_steps is injected at runtime.

    Unlike torch.optim.lr_scheduler.OneCycleLR, this class does not require
    total_steps at construction time. Instead, total_steps is passed when the
    scheduler is instantiated in configure_optimizers, where the trainer's
    estimated_stepping_batches is available.
    """

    def __init__(
        self,
        max_lr: float,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ) -> None:
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

    def __call__(self, optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.OneCycleLR:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
        )
