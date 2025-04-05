import pathlib
import comet_ml 

from lightning.pytorch.cli import ArgsType
from hepattn.utils.cli import CLI
from hepattn.experiments.itk.data import ITkDataModule
from hepattn.experiments.itk.model import ITkTracker



def main(args: ArgsType = None) -> None:
    CLI(
        model_class=ITkTracker,
        datamodule_class=ITkDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
