# hepattn/utils/loggers.py
from pathlib import Path
from lightning.pytorch.loggers import CometLogger

class MyCometLogger(CometLogger):
    """Wrap CometLogger to play nice with LightningCLI.SaveConfigCallback."""

    def __init__(
        self,
        name: str | None = None,                 # run/experiment name
        project_name: str | None = None,
        offline_directory: str | None = None,    # will be set by your CLI
        save_dir: str | None = None,             # accept either; we normalize
        log_env_details: bool = False,
        **kwargs,
    ):
        base = Path(offline_directory or save_dir or ".").resolve()
        base.mkdir(parents=True, exist_ok=True)
        self._save_dir = str(base)
        self._log_dir = self._save_dir

        # Comet uses 'experiment_name' (Lightning passes through)
        super().__init__(
            project_name=project_name,
            experiment_name=name,
            save_dir=self._save_dir,             # <-- important for Trainer.log_dir
            offline_directory=self._save_dir,    # <-- important for offline mode
            log_env_details=log_env_details,
            **kwargs,
        )

    @property
    def save_dir(self) -> str:   # what Trainer.log_dir typically reads
        return self._save_dir

    @property
    def log_dir(self) -> str:    # what LightningCLI.SaveConfigCallback may read
        return self._log_dir
