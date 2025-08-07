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


def _export_batched_pipeline_responses(
    all_detection_responses: list,
    occurrence_dicts: list,
    pipeline_slug: str,
    include_category_maps: bool,
    batch_size: Optional[int],
    images_per_batch: int,
    outfile: Optional[pathlib.Path],
    destination_dir: pathlib.Path,
) -> str:
    """
    Export pipeline responses split into multiple JSON files.

    Args:
        all_detection_responses: All detection responses to split
        occurrence_dicts: All occurrence dictionaries
        pipeline_slug: Pipeline name
        include_category_maps: Whether to include category maps
        batch_size: Number of detections per batch (takes precedence)
        images_per_batch: Number of source images per batch
        outfile: Output file path (used for naming pattern)
        destination_dir: Directory to save files

    Returns:
        String describing the export results
    """
    import json
    from collections import defaultdict

    # Group detections by source image
    detections_by_image = defaultdict(list)
    for detection in all_detection_responses:
        detections_by_image[detection.source_image_id].append(detection)

    # Group occurrences by source image for consistency
    occurrences_by_image = defaultdict(list)
    for occurrence in occurrence_dicts:
        for example in occurrence.get("examples", []):
            source_image_id = str(example.get("source_image_id", "unknown"))
            occurrences_by_image[source_image_id].append(occurrence)

    # Create batches
    batches = []
    if batch_size is not None:
        # Batch by number of detections
        current_batch_detections = []
        current_batch_occurrences = []

        for detection in all_detection_responses:
            current_batch_detections.append(detection)

            # Find corresponding occurrences for this detection
            source_image_id = detection.source_image_id
            for occurrence in occurrences_by_image[source_image_id]:
                if occurrence not in current_batch_occurrences:
                    current_batch_occurrences.append(occurrence)

            if len(current_batch_detections) >= batch_size:
                batches.append((current_batch_detections, current_batch_occurrences))
                current_batch_detections = []
                current_batch_occurrences = []

        # Add remaining detections as final batch
        if current_batch_detections:
            batches.append((current_batch_detections, current_batch_occurrences))
    else:
        # Batch by number of source images
        source_image_ids = list(detections_by_image.keys())

        for i in range(0, len(source_image_ids), images_per_batch):
            batch_image_ids = source_image_ids[i : i + images_per_batch]
            batch_detections = []
            batch_occurrences = []

            for image_id in batch_image_ids:
                batch_detections.extend(detections_by_image[image_id])
                batch_occurrences.extend(occurrences_by_image[image_id])

            # Remove duplicate occurrences
            unique_occurrences = []
            seen_occurrence_ids = set()
            for occurrence in batch_occurrences:
                occ_id = occurrence.get("id")
                if occ_id not in seen_occurrence_ids:
                    unique_occurrences.append(occurrence)
                    seen_occurrence_ids.add(occ_id)

            batches.append((batch_detections, unique_occurrences))

    # Export each batch
    exported_files = []
    timestamp = int(time.time())

    for batch_idx, (batch_detections, batch_occurrences) in enumerate(batches):
        # Create pipeline response for this batch
        pipeline_response = create_pipeline_results_response(
            occurrences=batch_occurrences,
            detection_responses=batch_detections,
            pipeline_name=pipeline_slug,
            total_time=0.0,
            include_category_maps=include_category_maps,
        )

        # Determine output filename
        if outfile:
            base_name = outfile.stem
            suffix = outfile.suffix
            batch_filename = f"{base_name}_batch_{batch_idx + 1:03d}{suffix}"
        else:
            batch_filename = (
                f"api_occurrences_{timestamp}_batch_{batch_idx + 1:03d}.json"
            )

        batch_filepath = destination_dir / batch_filename

        # Write batch file
        pipeline_dict = pipeline_response.model_dump()
        with open(batch_filepath, "w") as f:
            json.dump(pipeline_dict, f, indent=2, default=str)

        exported_files.append(str(batch_filepath.absolute()))

        logger.info(
            f"Exported batch {batch_idx + 1}/{len(batches)} with "
            f"{len(batch_detections)} detections from "
            f"{len({d.source_image_id for d in batch_detections})} source images "
            f'to "{batch_filepath}"'
        )

    summary = (
        f"Exported {len(all_detection_responses)} total detections across "
        f"{len(batches)} batch files:\n" + "\n".join(f"  - {f}" for f in exported_files)
    )

    logger.info(f"Batch export complete: {len(batches)} files created")
    return summary


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
    pipeline_slug: str,
    format: ExportFormat = ExportFormat.json,
    num_examples: int = 3,
    limit: Optional[int] = None,
    offset: int = 0,
    outfile: Optional[pathlib.Path] = None,
    collect_images: bool = False,
    absolute_paths: bool = False,
    detection_algorithm: Optional[str] = None,
    classification_algorithm: Optional[str] = None,
    include_category_maps: bool = False,
    images_per_batch: int = 100,
) -> Optional[str]:
    """
    Export occurrences using API schemas (DetectionResponse/ClassificationResponse).

    This exports the same occurrence data as the 'occurrences' command but uses
    the new API schema format with DetectionResponse and ClassificationResponse
    objects instead of the legacy Occurrence and ExportedDetection formats.

    Args:
        pipeline_slug: The pipeline reference in Antenna, must be one of the valid
                       choices from CLASSIFIER_CHOICES.
        images_per_batch: Number of source images per exported file (default: 100)
    """
    # Validate pipeline choice
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

    # Handle export based on format
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
        # Always use batching with default of 1 image per batch
        return _export_batched_pipeline_responses(
            all_detection_responses=all_detection_responses,
            occurrence_dicts=occurrence_dicts,
            pipeline_slug=pipeline_slug,
            include_category_maps=include_category_maps,
            batch_size=None,
            images_per_batch=images_per_batch,
            outfile=outfile,
            destination_dir=destination_dir,
        )
