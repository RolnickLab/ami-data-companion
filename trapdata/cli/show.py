import datetime
from typing import Type, Optional, Iterable, List

import typer
from rich import print
from sqlalchemy import select

from trapdata.db.base import get_session_class
from trapdata.db.models.detections import get_unique_species_by_track
from trapdata.db.models.events import (
    MonitoringSession,
    get_monitoring_session_by_date,
    get_monitoring_sessions_from_db,
)
from trapdata.settings import settings
from trapdata import logger

cli = typer.Typer()


@cli.command()
def events():
    """"""
    Session = get_session_class(settings.database_url)
    session = Session()
    events = session.execute(select(MonitoringSession)).unique().scalars().all()
    for event in events:
        print(event)


if __name__ == "__main__":
    cli()
