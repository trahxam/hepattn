from lightning.pytorch.cli import ArgsType
from hepattn.utils.cli import CLI
from hepattn.experiments.trackml.data import TrackMLDataModule
from hepattn.experiments.trackml.tracker import TrackMLTracker


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLTracker,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
