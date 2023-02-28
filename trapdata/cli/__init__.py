from trapdata.settings import read_settings


settings = read_settings()


if __name__ == "__main__":
    from trapdata.cli.base import cli

    cli()
