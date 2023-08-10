# import newrelic.agent
# newrelic.agent.initialize(environment="staging")

import json
import os
import pathlib
import tempfile
from typing import Union

import torch
from rich import print

from trapdata import logger
from trapdata.common.schemas import FilePath
from trapdata.db import check_db, get_session_class
from trapdata.db.models.events import get_or_create_monitoring_sessions
from trapdata.db.models.queue import (
    add_monitoring_session_to_queue,
    clear_all_queues,
    images_in_queue,
)
from trapdata.ml.models import (
    BinaryClassifierChoice,
    FeatureExtractorChoice,
    ObjectDetectorChoice,
    SpeciesClassifierChoice,
)
from trapdata.ml.models.tracking import summarize_tracks
from trapdata.ml.pipeline import start_pipeline
from trapdata.ml.utils import StopWatch
from trapdata.settings import PipelineSettings

# @newrelic.agent.background_task()


def get_settings(db_path: str, image_base_path: FilePath) -> PipelineSettings:
    settings = PipelineSettings(
        database_url=db_path,
        image_base_path=image_base_path,
        # user_data_path=pathlib.Path(tempfile.TemporaryDirectory(prefix="AMI-").name),
        localization_model=ObjectDetectorChoice.fasterrcnn_mobilenet_for_ami_moth_traps_2023,
        binary_classification_model=BinaryClassifierChoice.moth_nonmoth_classifier,
        species_classification_model=SpeciesClassifierChoice.quebec_vermont_species_classifier,
        feature_extractor=FeatureExtractorChoice.features_from_quebecvermont_species_model,
        classification_threshold=0.6,
        localization_batch_size=1,
        classification_batch_size=10,
        num_workers=1,
    )
    return settings


def setup_db(settings: PipelineSettings):
    check_db(settings.database_url, create=True)


def add_images(settings: PipelineSettings):
    # db_path = ":memory:"

    events = get_or_create_monitoring_sessions(
        settings.database_url, settings.image_base_path
    )

    clear_all_queues(settings.database_url, settings.image_base_path)
    # add_sample_to_queue(db_path, sample_size=sample_size)

    for event in events:
        add_monitoring_session_to_queue(settings.database_url, monitoring_session=event)

    num_images = images_in_queue(settings.database_url)
    logger.info(f"Images in queue: {num_images}")


def process_images(settings: PipelineSettings):
    Session = get_session_class(db_path=settings.database_url)
    session = Session()
    start_pipeline(
        session=session, image_base_path=settings.image_base_path, settings=settings
    )


def get_summary(settings: PipelineSettings) -> dict[Union[str, None], list[dict]]:
    # @TODO use list_occurrences instead
    Session = get_session_class(db_path=settings.database_url)
    session = Session()
    tracks = summarize_tracks(session=session)
    return tracks


def simple_results(tracks):
    items = []
    for sequence_id, track in tracks.items():
        items += track

    results = sorted(
        [(item["event"], item["sequence"], item["specific_label"]) for item in items]
    )
    print("Comparing:", results)
    return results


def compare_results(deployment_name: str, results: dict, expected_results: dict):
    # @TODO implement pytest and use list_occurrences() instead of summarize_tracks()
    assert simple_results(results) == simple_results(
        expected_results
    ), f"The pipeline returned different results than expected for the deployment '{deployment_name}'"


def process_deployment(deployment_subdir="vermont"):
    tests_dir = pathlib.Path(__file__).parent
    image_base_path = tests_dir / "images" / deployment_subdir
    logger.info(f"Using test images from: {image_base_path}")
    expected_results_path = tests_dir / "results" / f"{deployment_subdir}.json"

    local_weights_path = torch.hub.get_dir()
    logger.info(f"Looking for or downloading weights in {local_weights_path}")
    os.environ["LOCAL_WEIGHTS_PATH"] = local_weights_path

    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as db_filepath:
        db_path = f"sqlite+pysqlite:///{db_filepath.name}"
        logger.info(f"Using temporary DB: {db_path}")

        settings = get_settings(db_path, image_base_path)
        setup_db(settings)

        with StopWatch() as t:
            add_images(settings)
        logger.info(t)

        with StopWatch() as t:
            process_images(settings)
        logger.info(t)

        summary = get_summary(settings)

        if expected_results_path.exists():
            results = json.loads(json.dumps(summary, indent=2, default=str))
            expected_results = json.load(open(expected_results_path))
            compare_results(deployment_subdir, results, expected_results)
        else:
            print("Saving new results to", expected_results_path)
            json.dump(summary, open(expected_results_path, "w"), indent=2, default=str)


# def test_feature_extractor():
#     objects = (
#         session.execute(
#             select(DetectedObject).where(DetectedObject.cnn_features.is_not(None))
#         )
#         .unique()
#         .scalars()
#         .all()
#     )
#
#     for object in objects:
#         # logger.info(f"Number of features: {num_features}")
#         # assert (
#         #     num_features == 1536 * 10 * 10,
#         # )  # This is dependent on the input size & type of model
#         num_features = len(object.cnn_features)
#         assert (
#             num_features == 2048  # Num features expected for ResNet model
#         )  # This is dependent on the input size & type of model
#         result = cosine_similarity(object.cnn_features, object.cnn_features)
#         assert round(result, 1) == 1.0, "Cosine similarity of same object is not 1!"


def process_deployments():
    for deployment in ["vermont", "sequential"]:
        process_deployment(deployment)


if __name__ == "__main__":
    process_deployments()
