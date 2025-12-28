from lightning.pytorch.cli import ArgsType

from hepattn.experiments.top.data import TopEventDataModule
from hepattn.experiments.top.model import TopEventReconstructor
from hepattn.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TopEventReconstructor,
        datamodule_class=TopEventDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
