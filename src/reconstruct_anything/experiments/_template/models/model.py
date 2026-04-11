# ModelWrapper subclass with experiment-specific metrics.
# Define one file per model variant (e.g. filter.py, tracker.py).
# The config YAML selects which class to use via model.class_path.

from reconstruct_anything.models import ModelWrapper


class MyModel(ModelWrapper):
    def log_custom_metrics(self, preds, targets, stage):
        # Log experiment-specific metrics here.
        # Called automatically after each validation/test step.
        pass
