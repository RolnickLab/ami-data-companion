from trapdata.settings import read_settings
from trapdata.db import check_db


settings = read_settings()
check_db(settings.database_url, create=True, update=True)


if __name__ == "__main__":
    from trapdata.cli.base import cli

    cli()
