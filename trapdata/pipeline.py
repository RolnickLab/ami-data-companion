import pathlib

from trapdata import logger
from trapdata import ml
from trapdata.db.base import get_session_class
from trapdata.common.types import FilePath


def start_pipeline(
    db_path: str,
    image_base_path: FilePath,
    config: dict,
    single: bool = False,
):
    user_data_path = pathlib.Path(config.get("paths", "user_data_path"))
    logger.info(f"Local user data path: {user_data_path}")
    num_workers = int(config.get("performance", "num_workers"))

    object_detector_name = config.get("models", "localization_model")
    ObjectDetector = ml.models.object_detectors[object_detector_name]
    object_detector = ObjectDetector(
        db_path=db_path,
        image_base_path=image_base_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "localization_batch_size")),
        num_workers=num_workers,
        single=single,
    )
    if object_detector.queue.queue_count() > 0:
        object_detector.run()
        logger.info("Localization complete")

    binary_classifier_name = config.get("models", "binary_classification_model")
    BinaryClassifier = ml.models.binary_classifiers[binary_classifier_name]
    binary_classifier = BinaryClassifier(
        db_path=db_path,
        image_base_path=image_base_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "classification_batch_size")),
        num_workers=num_workers,
        single=single,
    )
    if binary_classifier.queue.queue_count() > 0:
        binary_classifier.run()
        logger.info("Binary classification complete")

    species_classifier_name = config.get("models", "species_classification_model")
    SpeciesClassifier = ml.models.species_classifiers[species_classifier_name]
    species_classifier = SpeciesClassifier(
        db_path=db_path,
        image_base_path=image_base_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "classification_batch_size")),
        num_workers=num_workers,
        single=single,
    )
    if species_classifier.queue.queue_count() > 0:
        species_classifier.run()
        logger.info("Species classification complete")

    feature_extractor_name = config.get("models", "feature_extractor")
    FeatureExtractor = ml.models.feature_extractors[feature_extractor_name]
    feature_extractor = FeatureExtractor(
        db_path=db_path,
        image_base_path=image_base_path,
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "classification_batch_size")),
        num_workers=num_workers,
        single=single,
    )
    if feature_extractor.queue.queue_count() > 0:
        feature_extractor.run()
        logger.info("Feature extraction complete")

    # @TODO this should only generate tracks with cnn_features for all detections
    # in a monitoring session. Consider creating or updating a QueueManager
    # instead of the code below.
    Session = get_session_class(db_path)
    with Session() as session:
        events = ml.models.tracking.get_events_that_need_tracks(
            base_directory=image_base_path,
            session=session,
        )
        for event in events:
            ml.models.tracking.find_all_tracks(
                monitoring_session=event, session=session
            )

        # Debug extra unprocessed objects:
        # from trapdata.ml.models.tracking import UntrackedObjectsQueue
        # queue = UntrackedObjectsQueue(db_path=db_path, base_directory=image_base_path)
        # queue.add_unprocessed()
        # for obj, previous_objects in queue.pull_n_from_queue(100):
        #     print(len(previous_objects), obj)
