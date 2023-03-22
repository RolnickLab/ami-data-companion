import typer
from rich import print
from rich.console import Console
from rich.table import Table
from sqlalchemy import select, func

from trapdata.db.base import get_session_class
from trapdata.cli import settings
from trapdata.db import models
from trapdata.db.models.detections import (
    num_occurrences_for_event,
    num_species_for_event,
)
from trapdata import logger

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
    deployments = session.execute(
        select(
            models.MonitoringSession.base_directory,
            func.count(models.MonitoringSession.id),
            func.sum(models.MonitoringSession.num_images),
            func.sum(models.MonitoringSession.num_detected_objects),
        ).group_by(models.MonitoringSession.base_directory)
    ).all()

    table = Table("Image Base Path", "Events", "Images", "Objects")
    for deployment in deployments:
        row_values = [str(field) for field in deployment._mapping.values()]
        table.add_row(*row_values)

    console.print(table)


@cli.command()
def events():
    """
    Show all monitoring events that have been interpreted from image timestamps.
    """
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

    table = Table("ID", "Day", "Images", "Detections", "Occurrences", "Species")
    for event in events:
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


if __name__ == "__main__":
    cli()
