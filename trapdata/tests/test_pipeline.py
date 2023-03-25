# import newrelic.agent
# newrelic.agent.initialize(environment="staging")

import os
import tempfile
import pathlib

import torch
from rich import print

from trapdata import logger
from trapdata.common.types import FilePath
from trapdata.settings import PipelineSettings
from trapdata.db import check_db, get_session_class
from trapdata.db.models.events import get_or_create_monitoring_sessions
from trapdata.db.models.queue import (
    add_sample_to_queue,
    add_monitoring_session_to_queue,
    images_in_queue,
    clear_all_queues,
)
from trapdata.ml.models.tracking import summarize_tracks

from trapdata.ml.utils import StopWatch
from trapdata.ml.models import (
    ObjectDetectorChoice,
    BinaryClassifierChoice,
    SpeciesClassifierChoice,
    FeatureExtractorChoice,
)

from trapdata.ml.pipeline import start_pipeline


# @newrelic.agent.background_task()


def get_settings(db_path: str, image_base_path: FilePath) -> PipelineSettings:
    settings = PipelineSettings(
        database_url=db_path,
        image_base_path=image_base_path,
        localization_model=ObjectDetectorChoice.fasterrcnn_for_ami_moth_traps,
        binary_classification_model=BinaryClassifierChoice.moth_nonmoth_classifier,
        species_classification_model=SpeciesClassifierChoice.quebec_vermont_species_classifier_mixed_resolution,
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


def show_summary(settings: PipelineSettings):
    Session = get_session_class(db_path=settings.database_url)
    session = Session()
    print(summarize_tracks(session=session))


def run():
    deployment_sub_dir = "vermont"
    image_base_path = pathlib.Path(__file__).parent / "images" / deployment_sub_dir
    logger.info(f"Using test images from: {image_base_path}")

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

        show_summary(settings)


if __name__ == "__main__":
    run()
