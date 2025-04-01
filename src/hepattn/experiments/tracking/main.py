import pathlib
import comet_ml 

from lightning.pytorch.cli import ArgsType
from hepattn.utils.cli import CLI
from hepattn.experiments.tracking.data import TrackingDataModule
from hepattn.experiments.tracking.tracker import Tracker



def main(args: ArgsType = None) -> None:
    CLI(
        model_class=Tracker,
        datamodule_class=TrackingDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
