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
from trapdata.api.export_utils import (
    convert_occurrence_to_detection_responses,
    create_pipeline_results_response,
)
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
from trapdata.db.models.occurrences import Occurrence, list_occurrences

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

    occurrences: list[Occurrence] = []

    tabular_formats = [ExportFormat.csv]
    plain_text_formats = [ExportFormat.csv, ExportFormat.html]
    if format in tabular_formats:
        num_examples = 1

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

    if outfile:
        destination_dir = outfile.parent
    else:
        destination_dir = settings.user_data_path / "exports"
    destination_dir.mkdir(parents=True, exist_ok=True)

    if collect_images:
        # Collect images for exported occurrences into a subdirectory
        if outfile:
            name = outfile.stem
        else:
            name = f"occurrences_{int(time.time())}"
        destination_dir = destination_dir / f"{name}_images"
        logger.info(f'Collecting images into "{destination_dir}"')
        destination_dir.mkdir(parents=True, exist_ok=True)

        for occurrence in occurrences:
            for example in occurrence.examples:
                path = pathlib.Path(example["cropped_image_path"]).resolve()
                destination = (
                    destination_dir / f"{occurrence.label} {occurrence.id} {path.name}"
                )
                if not destination.exists():
                    shutil.copy(path, destination)
                path = destination
                if absolute_paths:
                    final_path = path.absolute()
                else:
                    final_path = path.relative_to(destination_dir)
                example["cropped_image_path"] = final_path

    if format in tabular_formats:
        for occurrence in occurrences:
            if occurrence.examples:
                example = occurrence.examples[0]
                occurrence.example_crop = example["cropped_image_path"]
                occurrence.examples = []

    df = pd.DataFrame([obj.model_dump() for obj in occurrences])
    if format in tabular_formats:
        df = df.drop(columns=["examples"])
    if format in plain_text_formats:
        # df["cropped_image_path"] = df["cropped_image_path"].astype(str)
        # df["timestamp"] = df["timestamp"].astype(str)
        print(df.columns)

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
    df = pd.DataFrame([obj.report_data().model_dump() for obj in objects])
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
            img.report_data().model_dump() for img in example_captures
        ]
        event_data["num_occurrences"] = num_occurrences
        event_data["num_species"] = num_species
        items.append(event_data)
    df = pd.DataFrame(items)
    return export(df=df, format=format, outfile=outfile)


@cli.command()
def captures(
    date: Optional[datetime.datetime] = None,
    format: ExportFormat = ExportFormat.json,
    outfile: Optional[pathlib.Path] = None,
) -> Optional[str]:
    """
    List of source images for a given monitoring session.

    Date should be in the format YYYY-MM-DD
    """
    Session = get_session_class(settings.database_url)
    session = Session()
    if date is not None:
        event_dates = [date.date()]
    else:
        event_dates = [
            event.day
            for event in get_monitoring_sessions_from_db(
                db_path=settings.database_url, base_directory=settings.image_base_path
            )
        ]
    events = get_monitoring_session_by_date(
        db_path=settings.database_url,
        base_directory=settings.image_base_path,
        event_dates=event_dates,
    )
    if date and not len(events):
        raise Exception(f"No Monitoring Event with date: {date.date()}")

    captures = []
    for event in events:
        captures += get_monitoring_session_images(
            settings.database_url, event, limit=100
        )
    [session.add(img) for img in captures]

    df = pd.DataFrame([img.report_detail().model_dump() for img in captures])
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

    df = pd.DataFrame([d.model_dump() for d in deployments])
    return export(df=df, format=format, outfile=outfile)


@cli.command(name="api-occurrences")
def api_occurrences(
    format: ExportFormat = ExportFormat.json,
    num_examples: int = 3,
    limit: Optional[int] = None,
    offset: int = 0,
    outfile: Optional[pathlib.Path] = None,
    collect_images: bool = False,
    absolute_paths: bool = False,
    detection_algorithm: Optional[str] = None,
    classification_algorithm: Optional[str] = None,
) -> Optional[str]:
    """
    Export occurrences using API schemas (DetectionResponse/ClassificationResponse).

    This exports the same occurrence data as the 'occurrences' command but uses
    the new API schema format with DetectionResponse and ClassificationResponse
    objects instead of the legacy Occurrence and ExportedDetection formats.
    """
    events = get_monitoring_sessions_from_db(
        db_path=settings.database_url, base_directory=settings.image_base_path
    )

    # Get occurrence data using existing logic
    occurrences: list[Occurrence] = []
    tabular_formats = [ExportFormat.csv]

    if format in tabular_formats:
        num_examples = 1

    for event in events:
        occurrences += list_occurrences(
            settings.database_url,
            monitoring_session=event,
            classification_threshold=settings.classification_threshold,
            num_examples=num_examples,
            limit=limit,
            offset=offset,
        )

    # Convert occurrences to DetectionResponse objects
    all_detection_responses = []
    occurrence_dicts = []
    for occurrence in occurrences:
        occurrence_dict = occurrence.model_dump()
        occurrence_dicts.append(occurrence_dict)
        detection_responses = convert_occurrence_to_detection_responses(
            occurrence_dict,
            detection_algorithm_name=detection_algorithm,
            classification_algorithm_name=classification_algorithm,
        )
        all_detection_responses.extend(detection_responses)

    # Create full pipeline results response
    pipeline_response = create_pipeline_results_response(
        occurrences=occurrence_dicts,
        detection_responses=all_detection_responses,
        pipeline_name="local_batch_processor",
        total_time=0.0,
    )

    logger.info(
        f"Preparing to export pipeline response with {len(all_detection_responses)} detection records as {format}"
    )

    if outfile:
        destination_dir = outfile.parent
    else:
        destination_dir = settings.user_data_path / "exports"
    destination_dir.mkdir(parents=True, exist_ok=True)

    if collect_images:
        # Collect images for exported detections into a subdirectory
        if outfile:
            name = outfile.stem
        else:
            name = f"api_occurrences_{int(time.time())}"
        destination_dir = destination_dir / f"{name}_images"
        logger.info(f'Collecting images into "{destination_dir}"')
        destination_dir.mkdir(parents=True, exist_ok=True)

        for detection in all_detection_responses:
            if detection.crop_image_url:
                source_path = pathlib.Path(detection.crop_image_url).resolve()
                if source_path.exists():
                    # Create a meaningful filename
                    classification = "unknown"
                    if detection.classifications:
                        classification = detection.classifications[0].classification

                    destination = (
                        destination_dir
                        / f"{classification}_{detection.source_image_id}_{source_path.name}"
                    )
                    if not destination.exists():
                        shutil.copy(source_path, destination)

                    # Update the crop_image_url to point to the collected image
                    if absolute_paths:
                        detection.crop_image_url = str(destination.absolute())
                    else:
                        detection.crop_image_url = str(
                            destination.relative_to(destination_dir)
                        )

    # Convert to DataFrame for export based on format
    if format in tabular_formats:
        # For CSV, flatten the detection responses structure
        detection_dicts = [
            detection.model_dump() for detection in all_detection_responses
        ]
        flattened_dicts = []
        for detection_dict in detection_dicts:
            flat_dict = {
                "source_image_id": detection_dict["source_image_id"],
                "bbox_x1": detection_dict["bbox"]["x1"],
                "bbox_y1": detection_dict["bbox"]["y1"],
                "bbox_x2": detection_dict["bbox"]["x2"],
                "bbox_y2": detection_dict["bbox"]["y2"],
                "timestamp": detection_dict["timestamp"],
                "crop_image_url": detection_dict.get("crop_image_url"),
                "detection_algorithm_name": detection_dict["algorithm"]["name"],
                "detection_algorithm_key": detection_dict["algorithm"]["key"],
            }

            # Add classification data if available
            if detection_dict["classifications"]:
                classification = detection_dict["classifications"][0]
                flat_dict.update(
                    {
                        "classification": classification["classification"],
                        "classification_score": (
                            classification["scores"][0]
                            if classification["scores"]
                            else None
                        ),
                        "classification_algorithm_name": classification["algorithm"][
                            "name"
                        ],
                        "classification_algorithm_key": classification["algorithm"][
                            "key"
                        ],
                        "classification_timestamp": classification["timestamp"],
                    }
                )
            else:
                flat_dict.update(
                    {
                        "classification": None,
                        "classification_score": None,
                        "classification_algorithm_name": None,
                        "classification_algorithm_key": None,
                        "classification_timestamp": None,
                    }
                )

            flattened_dicts.append(flat_dict)

        df = pd.DataFrame(flattened_dicts)
        return export(df=df, format=format, outfile=outfile)
    else:
        # For JSON/HTML, export the full pipeline response directly
        import json

        pipeline_dict = pipeline_response.model_dump()

        if outfile:
            with open(outfile, "w") as f:
                json.dump(pipeline_dict, f, indent=2, default=str)
            logger.info(f'Exported pipeline response to "{outfile}"')
            return str(outfile.absolute())
        else:
            output = json.dumps(pipeline_dict, indent=2, default=str)
            print(output)
            return output
