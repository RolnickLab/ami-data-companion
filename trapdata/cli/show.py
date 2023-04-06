import typer
from rich import print
from rich.console import Console
from rich.live import Live
from rich.table import Table
from sqlalchemy import func, select

from trapdata import logger
from trapdata.cli import settings
from trapdata.db import models
from trapdata.db.base import get_session_class
from trapdata.db.models.detections import (
    num_occurrences_for_event,
    num_species_for_event,
)
from trapdata.db.models.events import (
    get_monitoring_sessions_from_db,
    update_all_aggregates,
)
from trapdata.db.models.queue import all_queues

cli = typer.Typer(no_args_is_help=True)

console = Console()


@cli.command(name="settings")
def show_settings():
    """
    Display the current settings that have been detected.
    """
    print(settings)


@cli.command()
def deployments():
    """
    List all image base directories that have been scanned.
    A proxy for "registered trap deployments".
    """
    Session = get_session_class(settings.database_url)
    session = Session()
    update_all_aggregates(session, settings.image_base_path)
    deployments = session.execute(
        select(
            models.MonitoringSession.base_directory,
            func.count(models.MonitoringSession.id),
            func.sum(models.MonitoringSession.num_images),
            func.sum(models.MonitoringSession.num_detected_objects),
        ).group_by(models.MonitoringSession.base_directory)
    ).all()

    table = Table("Image Base Path", "Events", "Images", "Objects")
    table.columns[0].overflow = "fold"
    for deployment in deployments:
        row_values = [str(field) for field in deployment._mapping.values()]
        table.add_row(*row_values)

    console.print(table)


@cli.command()
def captures(deployment: str):
    """
    Summarize the raw images captured by a deployment.
    """
    raise NotImplementedError


@cli.command()
def events():
    """
    Show all monitoring events that have been interpreted from image timestamps.
    """
    Session = get_session_class(settings.database_url)
    session = Session()
    # image_base_path = str(settings.image_base_path.resolve())

    update_all_aggregates(session, settings.image_base_path)
    logger.info(f"Show monitoring events for images in {settings.image_base_path}")
    events = (
        session.execute(
            select(models.MonitoringSession).where(
                models.MonitoringSession.base_directory == str(settings.image_base_path)
            )
        )
        .unique()
        .scalars()
        .all()
    )

    table = Table("ID", "Day", "Images", "Detections", "Occurrences", "Species")
    for event in events:
        event.update_aggregates(session)
        num_occurrences = num_occurrences_for_event(
            db_path=settings.database_url, monitoring_session=event
        )
        num_species = num_species_for_event(
            db_path=settings.database_url, monitoring_session=event
        )
        row_values = [
            event.id,
            event.day,
            event.num_images,
            event.num_detected_objects,
            num_occurrences,
            num_species,
        ]
        table.add_row(*[str(val) for val in row_values])
    console.print(table)


@cli.command()
def occurrences():
    events = get_monitoring_sessions_from_db(
        db_path=settings.database_url, base_directory=settings.image_base_path
    )
    occurrences: list[models.occurrences.Occurrence] = []
    for event in events:
        occurrences += models.occurrences.list_occurrences(settings.database_url, event)

    table = Table("Event", "Label", "Appearance", "Duration")
    for occurrence in occurrences:
        table.add_row(
            occurrence.event,
            occurrence.label,
            str(occurrence.start_time),
            str(occurrence.duration),
        )
    console.print(table)


def get_queue_table():
    table = Table("Queue", "Unprocessed", "Queued", "Done")
    for name, queue in all_queues(
        db_path=settings.database_url, base_directory=settings.image_base_path
    ).items():
        row_values = [
            queue.name,
            queue.unprocessed_count(),
            queue.queue_count(),
            queue.done_count(),
        ]
        table.add_row(*[str(val) for val in row_values])
    return table


@cli.command()
def queue():
    """
    Show counts waiting in each queue.
    """

    with Live(get_queue_table(), refresh_per_second=1) as live:
        while True:
            live.update(get_queue_table())

    # console.print(table)


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


if __name__ == "__main__":
    cli()
