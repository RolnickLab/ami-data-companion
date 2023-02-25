import pathlib

from trapdata import logger
from trapdata import ml
from trapdata.db.base import get_session_class
from trapdata.common.types import FilePath


def start_pipeline(
    db_path: str,
    deployment_path: FilePath,
    config: dict,
    single: bool = False,
):
    user_data_path = pathlib.Path(config.get("paths", "user_data_path"))
    logger.info(f"Local user data path: {user_data_path}")
    num_workers = int(config.get("performance", "num_workers"))

    model_1_name = config.get("models", "localization_model")
    Model_1 = ml.models.object_detectors[model_1_name]
    model_1 = Model_1(
        db_path=db_path,
        deployment_path=deployment_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "localization_batch_size")),
        num_workers=num_workers,
        single=single,
    )
    model_1.run()
    logger.info("Localization complete")

    model_2_name = config.get("models", "binary_classification_model")
    Model_2 = ml.models.binary_classifiers[model_2_name]
    model_2 = Model_2(
        db_path=db_path,
        deployment_path=deployment_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "classification_batch_size")),
        num_workers=num_workers,
        single=single,
    )
    model_2.run()
    logger.info("Binary classification complete")

    model_3_name = config.get("models", "taxon_classification_model")
    Model_3 = ml.models.species_classifiers[model_3_name]
    model_3 = Model_3(
        db_path=db_path,
        deployment_path=deployment_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "classification_batch_size")),
        num_workers=num_workers,
        single=single,
    )
    model_3.run()
    logger.info("Species classification complete")

    Model_4 = ml.models.tracking.FeatureExtractor
    model_4 = Model_4(
        db_path=db_path,
        deployment_path=deployment_path,
        cnn_features_model=model_3,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "classification_batch_size")),
        num_workers=num_workers,
        single=single,
    )
    model_4.run()

    Session = get_session_class(db_path)
    with Session() as session:
        events = ml.models.tracking.get_events_that_need_tracks(
            base_directory=deployment_path,
            session=session,
        )
        for event in events:
            ml.models.tracking.find_all_tracks(
                monitoring_session=event, session=session
            )
