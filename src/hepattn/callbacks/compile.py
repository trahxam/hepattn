import torch
from lightning import Callback


class Compile(Callback):
    """Callback that compiles the encoder and decoder sub-modules with ``torch.compile``.

    Compilation is deferred until the start of training or testing so that the
    first traced graph is created under the correct grad-mode, avoiding
    immediate recompiles after Lightning's sanity-check.

    Attributes:
        dynamic: Whether to use dynamic shapes during compilation.
        mode: Compilation mode passed to ``torch.compile`` (e.g. ``"reduce-overhead"``).
    """

    def __init__(
        self,
        dynamic: bool = True,
        mode: str | None = None,
        skip_dynamic_cudagraphs: bool = False,
    ):
        """Initialise compilation settings.

        Args:
            dynamic: Whether to enable dynamic shape compilation.
            mode: ``torch.compile`` mode string, or ``None`` for the default.
            skip_dynamic_cudagraphs: If ``True``, disables CUDA graph tracing for
                dynamic shapes via ``triton.cudagraph_skip_dynamic_graphs``.
        """
        super().__init__()
        self.dynamic = dynamic
        self.mode = mode
        self._compiled = False
        if skip_dynamic_cudagraphs:
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True  # noqa: SLF001

    def setup(self, trainer, pl_module, stage):
        """Store a reference to the trainer for use during compilation."""
        self.trainer = trainer

    def on_train_start(self, trainer, pl_module) -> None:
        """Compile the model at the start of training if not already compiled."""
        # Compile after Lightning's sanity check so the first traced graph is
        # created under the training grad-mode, avoiding immediate recompiles.
        if self._compiled:
            return
        self.trainer = trainer
        self.compile(pl_module)
        self._compiled = True

    def on_test_start(self, trainer, pl_module) -> None:
        """Compile the model at the start of testing if not already compiled."""
        if self._compiled:
            return
        self.trainer = trainer
        self.compile(pl_module)
        self._compiled = True

    def compile(self, module):
        """Compile the ``encoder`` and ``decoder`` children of ``module.model``."""
        model = dict(module.named_children()).get("model")
        if model is None:
            return
        if self.trainer.is_global_zero:
            print("-" * 80)
            print("compiling model...")
        for name, submodule in model.named_children():
            if name in {"encoder", "decoder"} and isinstance(submodule, torch.nn.Module):
                if self.trainer.is_global_zero:
                    print(f" -> compiling model.{name}...")
                submodule.compile(dynamic=self.dynamic, mode=self.mode)
        if self.trainer.is_global_zero:
            print("-" * 80, "\n")
