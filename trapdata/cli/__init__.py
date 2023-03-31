from trapdata.db import check_db
from trapdata.settings import read_settings

settings = read_settings()
check_db(settings.database_url, create=True, update=True, quiet=True)


if __name__ == "__main__":
    from trapdata.cli.base import cli

    cli()
