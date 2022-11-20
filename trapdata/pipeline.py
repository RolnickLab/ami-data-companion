import pathlib

from trapdata import logger
from trapdata import ml


def start_pipeline(config, single=False):

    user_data_path = pathlib.Path(config.get("paths", "user_data_path"))
    logger.info(f"Local user data path: {user_data_path}")
    num_workers = int(config.get("performance", "num_workers"))

    model_1_name = config.get("models", "localization_model")
    Model_1 = ml.models.object_detectors[model_1_name]
    model_1 = Model_1(
        db_path=config.get("paths", "database_url"),
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
        db_path=config.get("paths", "database_url"),
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
        db_path=config.get("paths", "database_url"),
        user_data_path=user_data_path,
        batch_size=int(config.get("performance", "classification_batch_size")),
        num_workers=num_workers,
        single=single,
    )
    model_3.run()
    logger.info("Species classification complete")
