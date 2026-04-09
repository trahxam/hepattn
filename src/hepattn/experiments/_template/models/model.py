# ModelWrapper subclass with experiment-specific metrics.
# Define one file per model variant (e.g. filter.py, tracker.py).
# The config YAML selects which class to use via model.class_path.

from torch import nn

from hepattn.models import ModelWrapper


class MyModel(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
    ):
        super().__init__(name, model, lrs_config, optimizer)

    def log_custom_metrics(self, preds, targets, stage):
        # Log experiment-specific metrics here.
        # Called automatically after each validation/test step.
        pass
