import datetime
from typing import Type, Optional, Iterable, List

import typer
from rich import print
import pandas as pd
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
def species_by_track(event_day: datetime.datetime):
    """"""
    Session = get_session_class(settings.database_url)
    session = Session()
    event = session.execute(
        select(MonitoringSession).where(
            # MonitoringSession.base_directory="",  @TODO retrieve from settings?
            MonitoringSession.day
            == event_day.date(),
        )
    ).scalar_one()
    print(f"Matched of event: {event}")
    get_unique_species_by_track(
        settings.database_url,
        monitoring_session=event,
        classification_threshold=0.1,
    )


if __name__ == "__main__":
    cli()
