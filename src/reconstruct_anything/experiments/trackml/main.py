# Entry point for training, validation, and testing.
# This file is the same for all experiments — the config YAML
# specifies which model and data classes to use via class_path.
#
# Usage:
#   python main.py fit --config configs/base.yaml
#   python main.py test --config configs/base.yaml --ckpt_path path/to/checkpoint.ckpt

from reconstruct_anything.utils.cli import CLI


def main(args=None):
    CLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
