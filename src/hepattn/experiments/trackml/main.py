import pathlib
import comet_ml 

from lightning.pytorch.cli import ArgsType
from hepattn.utils.cli import CLI
from hepattn.experiments.trackml.data import TrackMLDataModule
from hepattn.experiments.trackml.model import TrackMLTracker



def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLTracker,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
