import sys
import enum
import pathlib
import csv
from typing import Optional, Union, Type

import typer
from rich import print
import pandas as pd

from trapdata.db.base import get_session_class
from trapdata.db.models.detections import get_detected_objects
from trapdata.db.models.events import get_monitoring_sessions_from_db
from trapdata.ml.models import species_classifiers, SpeciesClassifier
from trapdata.ml.models.base import InferenceBaseClass
from trapdata.ml.models.tracking import find_all_tracks, summarize_tracks
from trapdata.settings import settings
from trapdata import logger

cli = typer.Typer()


@cli.command()
def run():
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
    assert cnn_model is not None
    session = Session()
    events = get_monitoring_sessions_from_db(db_path=settings.database_url)
    for event in events:
        find_all_tracks(
            monitoring_session=event,
            cnn_model=cnn_model,
            session=session,
        )
    print(summarize_tracks(session))


if __name__ == "__main__":
    cli()
