import torch

from reconstruct_anything.utils.cli import CLI

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")


def main(args=None):
    CLI(subclass_mode_model=True, subclass_mode_data=True, args=args, parser_kwargs={"default_env": True})


if __name__ == "__main__":
    main()
