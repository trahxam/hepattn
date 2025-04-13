from lightning.pytorch.callbacks import Callback


class ParamCountLogger(Callback):
    def on_train_start(self, trainer, pl_module):
        params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        self.logger.log_hyperparams({"trainable_params": params})
