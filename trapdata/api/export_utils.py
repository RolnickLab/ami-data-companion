"""
Utilities for converting database models to API schemas for export functionality.
"""

import datetime
from typing import Optional, Protocol

from trapdata import ml
from trapdata.api.schemas import (
    AlgorithmConfigResponse,
    AlgorithmReference,
    BoundingBox,
    ClassificationResponse,
    DetectionResponse,
    PipelineResultsResponse,
    SourceImageResponse,
)
from trapdata.settings import read_settings


class DetectedObjectLike(Protocol):
    """Protocol for objects that behave like DetectedObject for conversion."""

    id: Optional[int]
    specific_label: Optional[str]
    specific_label_score: Optional[float]
    bbox: Optional[list[int]]
    path: Optional[str]
    timestamp: Optional[datetime.datetime]
    detection_algorithm: Optional[str]
    classification_algorithm: Optional[str]


def create_algorithm_reference(
    algorithm_name: Optional[str], task_type: str = "detection"
) -> AlgorithmReference:
    """
    Create an AlgorithmReference from an algorithm name.

    Args:
        algorithm_name: Name of the algorithm, may be None for legacy data
        task_type: Type of task (detection, classification)

    Returns:
        AlgorithmReference object
    """
    if not algorithm_name:
        if task_type == "detection":
            algorithm_name = "unknown_detector"
            key = "unknown_detector"
        else:
            algorithm_name = "unknown_classifier"
            key = "unknown_classifier"
        return AlgorithmReference(name=algorithm_name, key=key)

    # Try to find the actual algorithm key from the model classes
    current_settings = read_settings()

    if task_type == "detection":
        detector_choice = current_settings.localization_model
        detector_class = ml.models.object_detectors.get(detector_choice.value)
        if detector_class and detector_class.name == algorithm_name:
            key = detector_class.get_key()
        else:
            # Fallback to generated key
            key = algorithm_name.lower().replace(" ", "_").replace("-", "_")
    else:
        # Check species classifier first
        species_choice = current_settings.species_classification_model
        species_class = ml.models.species_classifiers.get(species_choice.value)
        if species_class and species_class.name == algorithm_name:
            key = species_class.get_key()
        else:
            # Check binary classifier
            binary_choice = current_settings.binary_classification_model
            binary_class = ml.models.binary_classifiers.get(binary_choice.value)
            if binary_class and binary_class.name == algorithm_name:
                key = binary_class.get_key()
            else:
                # Fallback to generated key
                key = algorithm_name.lower().replace(" ", "_").replace("-", "_")

    return AlgorithmReference(name=algorithm_name, key=key)


def convert_classification_to_classification_response(
    detected_obj: DetectedObjectLike,
    algorithm_name: Optional[str] = None,
    timestamp: Optional[datetime.datetime] = None,
) -> ClassificationResponse:
    """
    Convert classification data from a DetectedObject to ClassificationResponse.

    Args:
        detected_obj: Database DetectedObject with classification data
        algorithm_name: Name of classification algorithm used
        timestamp: Timestamp for the classification

    Returns:
        ClassificationResponse object
    """
    if timestamp is None:
        timestamp = detected_obj.timestamp or datetime.datetime.now()

    # Use the specific label and score from the detected object
    classification = detected_obj.specific_label or "unknown"
    score = detected_obj.specific_label_score or 0.0

    # Create algorithm reference
    algorithm = create_algorithm_reference(
        algorithm_name or detected_obj.classification_algorithm,
        task_type="classification",
    )

    return ClassificationResponse(
        classification=classification,
        labels=None,  # Not available in database model
        scores=[score],  # Single score for the predicted class
        logits=[],  # Not stored in database
        inference_time=None,  # Not stored in database
        algorithm=algorithm,
        terminal=True,
        timestamp=timestamp,
    )


def convert_detected_object_to_detection_response(
    detected_obj: DetectedObjectLike,
    source_image_id: str,
    crop_image_url: Optional[str] = None,
    detection_algorithm_name: Optional[str] = None,
    classification_algorithm_name: Optional[str] = None,
) -> DetectionResponse:
    """
    Convert a DetectedObject from database to DetectionResponse API schema.

    Args:
        detected_obj: Database DetectedObject
        source_image_id: ID of the source image
        crop_image_url: URL to the cropped image (optional)
        detection_algorithm_name: Name of detection algorithm used
        classification_algorithm_name: Name of classification algorithm used

    Returns:
        DetectionResponse object with embedded ClassificationResponse
    """
    # Convert bounding box from list to BoundingBox object
    bbox_coords = detected_obj.bbox or [0, 0, 0, 0]
    # Convert int coordinates to float for BoundingBox
    bbox_coords_float = [float(coord) for coord in bbox_coords]
    bbox = BoundingBox.from_coords(bbox_coords_float)

    # Create detection algorithm reference
    detection_algorithm = create_algorithm_reference(
        detection_algorithm_name or detected_obj.detection_algorithm,
        task_type="detection",
    )

    # Create classification response if classification data exists
    classifications = []
    if detected_obj.specific_label:
        classification_response = convert_classification_to_classification_response(
            detected_obj,
            algorithm_name=classification_algorithm_name,
            timestamp=detected_obj.timestamp,
        )
        classifications.append(classification_response)

    # Use crop image path as URL if available
    if not crop_image_url and detected_obj.path:
        crop_image_url = str(detected_obj.path)

    return DetectionResponse(
        source_image_id=source_image_id,
        bbox=bbox,
        inference_time=None,  # Not stored in database
        algorithm=detection_algorithm,
        timestamp=detected_obj.timestamp or datetime.datetime.now(),
        crop_image_url=crop_image_url,
        classifications=classifications,
    )


def convert_occurrence_to_detection_responses(
    occurrence_data: dict,
    detection_algorithm_name: Optional[str] = None,
    classification_algorithm_name: Optional[str] = None,
) -> list[DetectionResponse]:
    """
    Convert occurrence data (with examples) to a list of DetectionResponse objects.

    Args:
        occurrence_data: Dictionary containing occurrence data with examples
        detection_algorithm_name: Name of detection algorithm used
        classification_algorithm_name: Name of classification algorithm used

    Returns:
        List of DetectionResponse objects
    """
    detection_responses = []

    # Get current algorithm names from settings if not provided
    if not detection_algorithm_name or not classification_algorithm_name:
        current_settings = read_settings()

        if not detection_algorithm_name:
            detector_choice = current_settings.localization_model
            detector_class = ml.models.object_detectors.get(detector_choice.value)
            if detector_class:
                detection_algorithm_name = detector_class.name

        if not classification_algorithm_name:
            species_choice = current_settings.species_classification_model
            species_class = ml.models.species_classifiers.get(species_choice.value)
            if species_class:
                classification_algorithm_name = species_class.name

    examples = occurrence_data.get("examples", [])
    for example in examples:
        # Create a mock DetectedObject from the example data
        class MockDetectedObject:
            def __init__(self, example_data):
                self.id = example_data.get("id")
                self.specific_label = example_data.get("label")
                self.specific_label_score = example_data.get("score")
                self.bbox = example_data.get("bbox", [0, 0, 0, 0])
                self.path = example_data.get("cropped_image_path")
                self.timestamp = example_data.get("timestamp")
                self.detection_algorithm = detection_algorithm_name
                self.classification_algorithm = classification_algorithm_name

        mock_obj = MockDetectedObject(example)
        source_image_id = str(example.get("source_image_id", "unknown"))

        detection_response = convert_detected_object_to_detection_response(
            mock_obj,
            source_image_id=source_image_id,
            detection_algorithm_name=detection_algorithm_name,
            classification_algorithm_name=classification_algorithm_name,
        )

        detection_responses.append(detection_response)

    return detection_responses


def get_current_algorithms() -> dict[str, AlgorithmConfigResponse]:
    """
    Get the currently configured algorithms from settings.

    Returns:
        Dictionary of algorithm configurations keyed by algorithm key
    """
    current_settings = read_settings()
    algorithms = {}

    # Get object detector
    detector_choice = current_settings.localization_model
    detector_class = ml.models.object_detectors.get(detector_choice.value)
    if detector_class:
        algorithms[detector_class.get_key()] = AlgorithmConfigResponse(
            name=detector_class.name,
            key=detector_class.get_key(),
            task_type="detection",
            description=getattr(detector_class, "description", None),
            version=1,
        )

    # Get binary classifier
    binary_choice = current_settings.binary_classification_model
    binary_class = ml.models.binary_classifiers.get(binary_choice.value)
    if binary_class:
        algorithms[binary_class.get_key()] = AlgorithmConfigResponse(
            name=binary_class.name,
            key=binary_class.get_key(),
            task_type="classification",
            description=getattr(binary_class, "description", None),
            version=1,
        )

    # Get species classifier
    species_choice = current_settings.species_classification_model
    species_class = ml.models.species_classifiers.get(species_choice.value)
    if species_class:
        algorithms[species_class.get_key()] = AlgorithmConfigResponse(
            name=species_class.name,
            key=species_class.get_key(),
            task_type="classification",
            description=getattr(species_class, "description", None),
            version=1,
        )

    return algorithms


def get_source_images_from_occurrences(occurrences: list) -> list[SourceImageResponse]:
    """
    Extract unique source images from occurrence data.

    Args:
        occurrences: List of occurrence dictionaries with examples

    Returns:
        List of SourceImageResponse objects
    """
    source_images = {}

    for occurrence in occurrences:
        examples = occurrence.get("examples", [])
        for example in examples:
            source_image_id = str(example.get("source_image_id", "unknown"))
            source_image_path = example.get("source_image_path", "")

            if source_image_id not in source_images:
                source_images[source_image_id] = SourceImageResponse(
                    id=source_image_id,
                    url=source_image_path,
                )

    return list(source_images.values())


def create_pipeline_results_response(
    occurrences: list,
    detection_responses: list[DetectionResponse],
    pipeline_name: str = "local_batch_processor",
    total_time: float = 0.0,
) -> PipelineResultsResponse:
    """
    Create a complete PipelineResultsResponse from occurrence data and responses.

    Args:
        occurrences: List of occurrence dictionaries
        detection_responses: List of DetectionResponse objects
        pipeline_name: Name of the pipeline used
        total_time: Total processing time

    Returns:
        Complete PipelineResultsResponse object
    """
    # Get current algorithms
    algorithms = get_current_algorithms()

    # Get source images
    source_images = get_source_images_from_occurrences(occurrences)

    return PipelineResultsResponse(
        pipeline=pipeline_name,
        algorithms=algorithms,
        total_time=total_time,
        source_images=source_images,
        detections=detection_responses,
    )
