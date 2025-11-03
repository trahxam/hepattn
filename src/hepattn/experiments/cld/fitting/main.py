from lightning.pytorch.cli import ArgsType

from hepattn.experiments.cld.fitting.data import CLDParticleDataModule
from hepattn.experiments.cld.fitting.model import CLDFitter
from hepattn.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=CLDFitter,
        datamodule_class=CLDParticleDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
