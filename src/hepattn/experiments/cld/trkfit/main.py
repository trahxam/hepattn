from lightning.pytorch.cli import ArgsType

from hepattn.experiments.cld.trkfit.data import CLDTrackDataModule
from hepattn.experiments.cld.trkfit.model import CLDTrackFitter
from hepattn.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=CLDTrackFitter,
        datamodule_class=CLDTrackDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
