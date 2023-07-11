import datetime
import pathlib
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from trapdata import logger, ml
from trapdata.cli import settings
from trapdata.cli.queue import status as queue_status
from trapdata.common.filemanagement import find_images
from trapdata.db import models
from trapdata.db.base import get_session_class
from trapdata.db.models.deployments import list_deployments
from trapdata.db.models.detections import get_detected_objects
from trapdata.db.models.events import (
    get_monitoring_session_by_date,
    update_all_aggregates,
)
from trapdata.db.models.occurrences import list_occurrences, list_species
from trapdata.ml.utils import StopWatch

cli = typer.Typer(no_args_is_help=True)

console = Console()


@cli.command(name="settings")
def show_settings():
    """
    Display the current settings that have been detected.
    """
    print(settings)


@cli.command(name="models")
def ml_models():
    """
    List all available models that have been registered.
    """
    for model_type, model_list in [
        ("Object Detectors", ml.models.object_detectors),
        ("Binary Classifiers", ml.models.binary_classifiers),
        ("Species Classifiers", ml.models.species_classifiers),
        ("Feature Extractors", ml.models.feature_extractors),
    ]:
        table = Table(
            "[green]Name[/green] / [yellow]Key[/yellow]",
            "Description",
            title=model_type,
            title_style="bold",
            title_justify="left",
        )
        table.columns[0].overflow = "fold"

        for model in model_list.values():
            name = f"[green]{model.name}[/green] \n[yellow]{model.get_key()}[/yellow]\n"
            table.add_row(name, model.description)

        console.print(table)


@cli.command()
def deployments():
    """
    List all image base directories that have been scanned.
    A proxy for "registered trap deployments".
    """
    Session = get_session_class(settings.database_url)
    session = Session()
    update_all_aggregates(session, settings.image_base_path)
    deployments = list_deployments(session)
    table = Table(
        "ID",
        "Name",
        "Image Base Path",
        "Sessions",
        "Images",
        "Detections",
    )
    table.columns[0].overflow = "fold"
    for deployment in deployments:
        row_values = [str(field) for field in deployment.dict().values()]
        table.add_row(*row_values)

    console.print(table)


@cli.command()
def captures(path: Optional[pathlib.Path] = None, max_num: Optional[int] = None):
    """
    Summarize the raw images captured by a deployment.
    """
    if not path:
        path = settings.image_base_path
    images = []
    i = 0
    with StopWatch() as t:
        for i, f in enumerate(find_images(path)):
            # logger.debug(f'Found {f["path"].name} from {f["timestamp"].strftime("%c")}')
            images.append(f)
            if max_num and i + 1 >= max_num:
                break

    print(f"Total images: {i+1}")
    print(t)
    print(images)


@cli.command()
def capture_counts(path: Optional[pathlib.Path] = None, check_exif: bool = False):
    """
    Show number of raw images captured by a deployment as fast as possible.

    By default does not read timestamps from image EXIF data.
    """
    if not path:
        path = settings.image_base_path
    with StopWatch() as t:
        count = sum(1 for _ in find_images(path, check_exif=check_exif))
    print(f"Total images: {count}")
    print(t)
    print(0)


@cli.command()
def sessions():
    """
    Show all monitoring events that have been interpreted from image timestamps.
    """
    from trapdata.db.models.events import list_monitoring_sessions

    Session = get_session_class(settings.database_url)
    session = Session()
    # image_base_path = str(settings.image_base_path.resolve())

    events = list_monitoring_sessions(session, settings.image_base_path)

    table = Table(
        "ID", "Day", "Duration", "Captures", "Detections", "Occurrences", "Species"
    )
    for event in events:
        row_values = [
            event.id,
            event.day,
            event.duration_label,
            event.num_captures,
            event.num_detections,
            event.num_occurrences,
            event.num_species,
        ]
        table.add_row(*[str(val) for val in row_values])
    console.print(table)


@cli.command()
def detections(
    session_day: Optional[datetime.datetime] = None,
    limit: Optional[int] = 100,
    offset: int = 0,
):
    """
    Show all detections for a given event.
    """
    if session_day:
        sessions = get_monitoring_session_by_date(
            db_path=settings.database_url,
            base_directory=settings.image_base_path,
            event_dates=[session_day],
        )
        session = sessions[0] if sessions else None
    else:
        session = None
    detections = get_detected_objects(
        db_path=settings.database_url,
        image_base_path=settings.image_base_path,
        monitoring_session=session,
        limit=limit,
        offset=offset,
        classification_threshold=settings.classification_threshold,
    )
    table = Table("Session", "Label", "Score", "Timestamp", "Sequence", "Frame")
    for detection in detections:
        table.add_row(
            str(detection.monitoring_session.day),
            detection.specific_label,
            str(
                round(detection.specific_label_score, 2)
                if detection.specific_label_score
                else ""
            ),
            str(detection.timestamp),
            detection.sequence_id,
            str(detection.sequence_frame),
        )
    console.print(table)


@cli.command()
def occurrences(
    session_day: Optional[datetime.datetime] = None,
    limit: Optional[int] = 100,
    offset: int = 0,
):
    event = None
    if session_day:
        events = get_monitoring_session_by_date(
            db_path=settings.database_url, event_dates=[session_day]
        )
        if not events:
            logger.info(f"No events found for {session_day}")
            return []
        else:
            event = events[0]

    occurrences = list_occurrences(
        settings.database_url,
        settings.image_base_path,
        event,
        classification_threshold=settings.classification_threshold,
        limit=limit,
        offset=offset,
    )

    table = Table(
        "Session", "ID", "Label", "Detections", "Score", "Appearance", "Duration"
    )
    for occurrence in occurrences:
        table.add_row(
            str(occurrence.event.day),
            occurrence.id,
            occurrence.label,
            str(occurrence.num_frames),
            str(round(occurrence.best_score, 2)),
            str(occurrence.start_time),
            str(occurrence.duration),
        )
    console.print(table)


@cli.command()
def species():
    table = Table("Name", "Count")
    for taxon in list_species(settings.database_url, settings.image_base_path):
        table.add_row(taxon.name, str(taxon.count))
    console.print(table)


@cli.command()
def missing_tracks():
    """ """
    Session = get_session_class(settings.database_url)
    session = Session()
    # image_base_path = str(settings.image_base_path.resolve())
    image_base_path = str(settings.image_base_path)
    logger.info(f"Show monitoring events for images in {image_base_path}")
    items = (
        session.execute(
            select(models.DetectedObject).where(
                (models.DetectedObject.sequence_id.is_(None))
                & (models.DetectedObject.specific_label.is_not(None))
            )
        )
        .unique()
        .scalars()
        .all()
    )
    print(items)
    print(len(items))


@cli.command()
def queue(watch: bool = False):
    """
    Show the status of the processing queue.

    This is an alias for `ami queue status`.
    """
    queue_status(watch)


if __name__ == "__main__":
    cli()
