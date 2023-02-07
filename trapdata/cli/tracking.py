import datetime
from typing import Type, Optional, Iterable, List

import typer
from rich import print
import pandas as pd
from sqlalchemy import select

from trapdata.db.base import get_session_class
from trapdata.db.models.detections import get_detected_objects
from trapdata.db.models.events import (
    MonitoringSession,
    get_monitoring_session_by_date,
    get_monitoring_sessions_from_db,
)
from trapdata.ml.models import species_classifiers, SpeciesClassifier
from trapdata.ml.models.base import InferenceBaseClass
from trapdata.ml.models.tracking import find_all_tracks, summarize_tracks
from trapdata.ml.utils import get_device
from trapdata.settings import settings
from trapdata import logger

cli = typer.Typer()


@cli.command()
def summary(event_day: datetime.datetime):
    Session = get_session_class(settings.database_url)
    session = Session()
    event = None
    if event_day:
        event = session.execute(
            select(MonitoringSession).where(
                # MonitoringSession.base_directory="",  @TODO retrieve from settings?
                MonitoringSession.day
                == event_day.date(),
            )
        ).scalar_one()
        print(f"Matched of event: {event}")
        print(summarize_tracks(session, event=event))


@cli.command()
def run(event_dates: List[datetime.datetime]):
    """
    Find tracks in all monitoring sessions .
    """
    Session = get_session_class(settings.database_url)
    CNNClassifier: Type[InferenceBaseClass] = species_classifiers[
        settings.species_classification_model.value
    ]
    logger.info(f"Using '{CNNClassifier}' to calculate CNN features")
    species_classifier = CNNClassifier(
        db_path=settings.database_url, user_data_path=settings.user_data_path
    )
    cnn_model = species_classifier.model
    device = get_device()
    assert cnn_model is not None
    session = Session()
    if event_dates:
        dates = [e.date() for e in event_dates]
        events = get_monitoring_session_by_date(
            db_path=settings.database_url, event_dates=dates
        )
    else:
        events = get_monitoring_sessions_from_db(db_path=settings.database_url)
    for event in events:
        find_all_tracks(
            monitoring_session=event,
            cnn_model=cnn_model,
            session=session,
            device=device,
        )


if __name__ == "__main__":
    cli()
