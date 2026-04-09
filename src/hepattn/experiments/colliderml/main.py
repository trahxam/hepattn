from hepattn.utils.cli import CLI


def main(args=None):
    CLI(subclass_mode_model=True, subclass_mode_data=True, args=args, parser_kwargs={"default_env": True})


if __name__ == "__main__":
    main()
