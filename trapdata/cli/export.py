import enum
import pathlib
import csv
from typing import Optional, Union

import typer
from rich import print
import pandas as pd

from trapdata import logger
from trapdata.cli import settings
from trapdata.db import get_session_class
from trapdata.db.models.detections import (
    get_detected_objects,
    num_occurrences_for_event,
    num_species_for_event,
)
from trapdata.db.models.events import get_monitoring_sessions_from_db
from trapdata.db.models.deployments import list_deployments
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
        return str(outfile.absolute())
    else:
        if output:
            # @TODO write the output to stdout without other log messages
            # sys.stdin.write(str(output))
            print(output)
        return output


@cli.command()
def occurrences(
    # deployment: Optional[str] = None,
    format: ExportFormat = ExportFormat.json,
    limit: Optional[int] = 10,
    offset: int = 0,
    outfile: Optional[pathlib.Path] = None,
) -> Optional[str]:
    """
    Export grouped occurrences from database in the specified format.
    """
    events = get_monitoring_sessions_from_db(
        db_path=settings.database_url, base_directory=settings.image_base_path
    )
    occurrences = []
    for event in events:
        occurrences += list_occurrences(settings.database_url, event)
    logger.info(f"Preparing to export {len(occurrences)} records as {format}")
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
def events(
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
        event_data["num_occurrences"] = num_occurrences
        event_data["num_species"] = num_species
        items.append(event_data)
    df = pd.DataFrame(items)
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

    df = pd.DataFrame(deployments)
    return export(df=df, format=format, outfile=outfile)


if __name__ == "__main__":
    cli()
