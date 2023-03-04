import typer
from rich import print
from sqlalchemy import select

from trapdata.db.base import get_session_class
from trapdata.cli import settings
from trapdata.db import models
from trapdata import logger

cli = typer.Typer(no_args_is_help=True)


@cli.command(name="settings")
def show_settings():
    """"""
    print(settings)


@cli.command()
def events():
    """"""
    Session = get_session_class(settings.database_url)
    session = Session()
    # image_base_path = str(settings.image_base_path.resolve())
    image_base_path = str(settings.image_base_path)
    logger.info(f"Show monitoring events for images in {image_base_path}")
    events = (
        session.execute(
            select(models.MonitoringSession).where(
                models.MonitoringSession.base_directory == image_base_path
            )
        )
        .unique()
        .scalars()
        .all()
    )
    for event in events:
        print(event)


if __name__ == "__main__":
    cli()
