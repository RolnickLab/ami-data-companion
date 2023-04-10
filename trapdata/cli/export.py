import csv
import datetime
import enum
import pathlib
import shutil
import time
from typing import Optional, Union

import pandas as pd
import typer
from rich import print

from trapdata import logger
from trapdata.cli import settings
from trapdata.db import get_session_class
from trapdata.db.models.deployments import list_deployments
from trapdata.db.models.detections import (
    get_detected_objects,
    num_occurrences_for_event,
    num_species_for_event,
)
from trapdata.db.models.events import (
    get_monitoring_session_by_date,
    get_monitoring_session_images,
    get_monitoring_sessions_from_db,
)
from trapdata.db.models.occurrences import list_occurrences

cli = typer.Typer(no_args_is_help=True)


class ExportFormat(str, enum.Enum):
    json = "json"
    html = "html"
    csv = "csv"


def export(
    df: pd.DataFrame,
    format: ExportFormat = ExportFormat.json,
    outfile: Optional[pathlib.Path] = None,
) -> Union[str, None]:
    df = df.convert_dtypes()
    if format is ExportFormat.json:
        output = df.to_json(
            path_or_buf=outfile,
            orient="records",
            indent=2,
            date_format="iso",
            default_handler=str,
        )
    elif format is ExportFormat.csv:
        output = df.to_csv(
            path_or_buf=outfile,
            quoting=csv.QUOTE_NONNUMERIC,
            index=False,
        )
    else:
        export_method = getattr(df, f"to_{format}")
        output = export_method(path_or_buf=outfile, index=False)
    if outfile:
        logger.info(f'Exported {len(df)} records to "{outfile}"')
        return str(outfile.absolute())
    if output:
        # @TODO write the output to stdout without other log messages
        # sys.stdin.write(str(output))
        print(output)
    return output


@cli.command()
def occurrences(
    # deployment: Optional[str] = None,
    format: ExportFormat = ExportFormat.json,
    num_examples: int = 3,
    limit: Optional[int] = None,
    offset: int = 0,
    outfile: Optional[pathlib.Path] = None,
    collect_images: bool = False,
    absolute_paths: bool = False,
    # create_zip: bool = False,  # @TODO write a zip file of the exported images with extended EXIF data
) -> Optional[str]:
    """
    Export detected occurrences from the active deployment / image_base_dir.

    @TODO nested examples in output do not work well with CSV format. Set num_examples to 1 as a workaround.
    """
    events = get_monitoring_sessions_from_db(
        db_path=settings.database_url, base_directory=settings.image_base_path
    )
    occurrences = []
    for event in events:
        occurrences += list_occurrences(
            settings.database_url,
            monitoring_session=event,
            classification_threshold=settings.classification_threshold,
            num_examples=num_examples,
            limit=limit,
            offset=offset,
        )
    logger.info(f"Preparing to export {len(occurrences)} records as {format}")

    if collect_images:
        # Collect images for exported occurrences into a subdirectory
        subdir = "exports"
        if outfile:
            name = outfile.stem
        else:
            name = f"occurrences_{int(time.time())}"
        destination_dir = settings.user_data_path / subdir / f"{name}_images"
        logger.info(f'Collecting images into "{destination_dir}"')
        destination_dir.mkdir(parents=True, exist_ok=True)
    else:
        destination_dir = None

    for occurrence in occurrences:
        for example in occurrence.examples:
            path = pathlib.Path(example["cropped_image_path"]).resolve()
            if destination_dir:
                destination = destination_dir / f"{occurrence.id}-{path.name}"
                if not destination.exists():
                    shutil.copy(path, destination)
                path = destination
            if absolute_paths:
                final_path = path.absolute()
            else:
                final_path = path.relative_to(settings.user_data_path)
            example["cropped_image_path"] = final_path

    df = pd.DataFrame([obj.dict() for obj in occurrences])
    return export(df=df, format=format, outfile=outfile)


@cli.command()
def detections(
    # deployment: Optional[str] = None,
    format: ExportFormat = ExportFormat.json,
    limit: Optional[int] = 10,
    offset: int = 0,
    outfile: Optional[pathlib.Path] = None,
) -> Optional[str]:
    """
    Export detected objects from database in the specified format.
    """
    objects = get_detected_objects(
        settings.database_url,
        limit=limit,
        offset=offset,
        image_base_path=settings.image_base_path,
    )
    logger.info(f"Preparing to export {len(objects)} records as {format}")
    df = pd.DataFrame([obj.report_data() for obj in objects])
    return export(df=df, format=format, outfile=outfile)


@cli.command()
def sessions(
    format: ExportFormat = ExportFormat.json,
    outfile: Optional[pathlib.Path] = None,
) -> Optional[str]:
    """
    Export a summary of monitoring sessions from database in the specified format.
    """
    monitoring_events = get_monitoring_sessions_from_db(
        db_path=settings.database_url, base_directory=settings.image_base_path
    )
    items = []
    for event in monitoring_events:
        event_data = event.report_data()
        num_occurrences = num_occurrences_for_event(
            db_path=settings.database_url, monitoring_session=event
        )
        num_species = num_species_for_event(
            db_path=settings.database_url, monitoring_session=event
        )
        example_captures = get_monitoring_session_images(
            settings.database_url, event, limit=5, offset=int(event.num_images / 2)
        )
        event_data["example_captures"] = [
            img.report_data().dict() for img in example_captures
        ]
        event_data["num_occurrences"] = num_occurrences
        event_data["num_species"] = num_species
        items.append(event_data)
    df = pd.DataFrame(items)
    return export(df=df, format=format, outfile=outfile)


@cli.command()
def captures(
    date: datetime.datetime,
    format: ExportFormat = ExportFormat.json,
    outfile: Optional[pathlib.Path] = None,
) -> Optional[str]:
    """
    List of source images for a given monitoring session.

    Date should be in the format YYYY-MM-DD
    """
    Session = get_session_class(settings.database_url)
    session = Session()
    events = get_monitoring_session_by_date(
        db_path=settings.database_url,
        base_directory=settings.image_base_path,
        event_dates=[str(date.date())],
    )
    if not len(events):
        raise Exception(f"No Monitoring Event with date: {date.date()}")

    event = events[0]
    captures = get_monitoring_session_images(settings.database_url, event, limit=100)
    [session.add(img) for img in captures]

    df = pd.DataFrame([img.report_detail().dict() for img in captures])
    return export(df=df, format=format, outfile=outfile)


@cli.command()
def deployments(
    format: ExportFormat = ExportFormat.json,
    outfile: Optional[pathlib.Path] = None,
) -> Optional[str]:
    """
    Export info about deployments inferred from image base directories.
    """
    Session = get_session_class(settings.database_url)
    session = Session()
    deployments = list_deployments(session)

    df = pd.DataFrame([d.dict() for d in deployments])
    return export(df=df, format=format, outfile=outfile)
