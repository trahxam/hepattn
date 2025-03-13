import torch
from lightning import Callback

from hepattn.utils.cuda_timer import cuda_timer


class InferenceTimer(Callback):
    def __init__(self):
        super().__init__()

    def on_test_start(self, trainer, pl_module):  # noqa: ARG002
        self.old_forward = pl_module.forward
        self.times = []

        def new_forward(*args, **kwargs):
            with cuda_timer(self.times):
                return self.old_forward(*args, **kwargs)

        pl_module.forward = new_forward

    def on_test_end(self, trainer, pl_module):  # noqa: ARG002
        pl_module.forward = self.old_forward
        self.times = self.times[5:]  # ensure warm start
        self.times = torch.tensor(self.times)
        self.mean_time = self.times.mean().item()
        self.std_time = self.times.std().item()

    def teardown(self, trainer, pl_module, stage):  # noqa: ARG002
        print("-" * 80)
        print(f"Mean inference time: {self.mean_time:.3f} ± {self.std_time:.3f} ms")
        print("-" * 80)

        # save full list of inference times
