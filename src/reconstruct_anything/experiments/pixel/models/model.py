from reconstruct_anything.models import ModelWrapper


class PixelClusterSplitter(ModelWrapper):
    def log_custom_metrics(self, preds, targets, stage):
        # Just log predictions from the final layer
        preds = preds["final"]
