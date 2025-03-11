import torch
from lightning import Callback


class Compile(Callback):
    def on_train_start(self, trainer, pl_module):  # noqa: ARG002
        self.compile(pl_module)

    def on_test_start(self, trainer, pl_module):  # noqa: ARG002
        self.compile(pl_module)

    def compile(self, pl_module):
        print("compiling encoder...")
        if hasattr(pl_module, "encoder"):
            pl_module.encoder = torch.compile(pl_module.encoder, dynamic=True)
        elif hasattr(pl_module.model, "encoder"):
            pl_module.model.encoder = torch.compile(pl_module.model.encoder, dynamic=True)
