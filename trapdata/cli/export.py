import sys
import enum
from typing import Optional

import typer
from rich import print
import pandas as pd

from trapdata.db.models.detections import get_detected_objects
from trapdata.settings import settings

cli = typer.Typer()


class ExportFormat(str, enum.Enum):
    json = "json"
    html = "html"
    csv = "csv"


@cli.command()
def detections(
    format: ExportFormat = ExportFormat.json,
    limit: Optional[int] = None,
    offset: int = 0,
) -> str:
    objects = get_detected_objects(settings.database_url, limit=limit, offset=offset)
    df = pd.DataFrame([obj.report_data() for obj in objects])
    if format is ExportFormat.json:
        output = df.to_json(orient="records", indent=2, default_handler=str)
    else:
        export_method = getattr(df, f"to_{format}")
        output = export_method(index=False)
    print(output)
    return output


if __name__ == "__main__":
    cli()
