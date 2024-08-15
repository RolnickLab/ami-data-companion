import typer
from trapdata import db
from trapdata.cli import settings

cli = typer.Typer(no_args_is_help=True)


@cli.command()
def create():
    """
    Create database tables and sqlite file if neccessary.
    """
    db.create_db(settings.database_url)
    db.migrate(settings.database_url)
    db.check_db(settings.database_url, quiet=False)


@cli.command()
def update():
    """
    Run database migrations to update the database schema.
    """
    db.migrate(settings.database_url)
    db.check_db(settings.database_url, quiet=False)


@cli.command()
def reset():
    """
    Backup and recreate database tables.
    """
    reset = typer.confirm("Are you sure you want to reset the database?")
    if reset:
        db.reset_db(settings.database_url)
    else:
        typer.Abort()


@cli.command()
def check():
    """
    Validate database tables and ORM models.
    """
    db.check_db(settings.database_url, quiet=False)


if __name__ == "__main__":
    cli()
