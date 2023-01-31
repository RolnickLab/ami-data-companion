import sys
import enum
import pathlib
import csv
from typing import Optional, Union

import typer
from rich import print
import pandas as pd

from trapdata.db.models.detections import get_detected_objects
from trapdata.db.models.events import get_monitoring_sessions_from_db
from trapdata.settings import settings
from trapdata import logger

cli = typer.Typer()


class ExportFormat(str, enum.Enum):
    json = "json"
    html = "html"
    csv = "csv"


def export(
    df: pd.DataFrame,
    format: ExportFormat = ExportFormat.json,
    outfile: Optional[pathlib.Path] = None,
) -> Union[str, None]:
    print(df.dtypes)
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
def detections(
    # trap: Optional[str] = None,
    format: ExportFormat = ExportFormat.json,
    limit: Optional[int] = 10,
    offset: int = 0,
    outfile: Optional[pathlib.Path] = None,
) -> Optional[str]:
    """
    Export detected objects from database in the specified format.

    Dates and
    """
    objects = get_detected_objects(settings.database_url, limit=limit, offset=offset)
    logger.info(f"Preparing to export {objects.count()} records as {format}")
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
    objects = get_monitoring_sessions_from_db(db_path=settings.database_url)
    df = pd.DataFrame([obj.report_data() for obj in objects])
    return export(df=df, format=format, outfile=outfile)


if __name__ == "__main__":
    cli()
