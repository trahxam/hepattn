import pathlib
import comet_ml 

from lightning.pytorch.cli import ArgsType
from hepattn.utils.cli import CLI
from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.model import CLDReconstructor


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=CLDReconstructor,
        datamodule_class=CLDDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
