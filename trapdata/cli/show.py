import typer
from rich import print
from sqlalchemy import select

from trapdata.db.base import get_session_class
from trapdata.settings import settings
from trapdata.db import models

cli = typer.Typer()


@cli.command(name="settings")
def show_settings():
    """"""
    print(settings)


@cli.command()
def events():
    """"""
    Session = get_session_class(settings.database_url)
    session = Session()
    events = session.execute(select(models.MonitoringSession)).unique().scalars().all()
    for event in events:
        print(event)


if __name__ == "__main__":
    cli()
