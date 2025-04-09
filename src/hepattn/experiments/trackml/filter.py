import torch.nn as nn

from hepattn.models.wrapper import ModelWrapper


class TrackMLFilter(ModelWrapper):
    def __init__(
            self,
            model: nn.Module,
            lrs_config: dict,
            optimizer: str = "AdamW",
        ):
        super().__init__(model, lrs_config, optimizer)
