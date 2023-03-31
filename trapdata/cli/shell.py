import typer

from trapdata.cli import settings
from trapdata.db.base import get_session_class

cli = typer.Typer(no_args_is_help=True)


@cli.command()
def ipython():
    """
    Open python shell with project loaded.
    """
    Session = get_session_class(settings.database_url)
    session = Session()  # noqa: F841
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    cli()
