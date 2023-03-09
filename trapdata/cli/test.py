import typer
from rich import print


from trapdata.cli import settings
from trapdata.tests import test_pipeline
from trapdata.db.base import check_db

cli = typer.Typer(no_args_is_help=True)


@cli.command()
def nothing():
    print("It works!")


@cli.command()
def pipeline():
    test_pipeline.run()


@cli.command()
def database():
    return check_db(db_path=settings.database_url, create=True, quiet=False)


if __name__ == "__main__":
    cli()
