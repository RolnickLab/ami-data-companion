import os
import json
import argparse
import pathlib
import shutil


import sys

# REPO_PATH = pathlib.Path(__file__).parent.parent

# @TODO Integrate classification into project
sys.path.append("/home/michael/Projects/AMI")
sys.path.append("/Users/michael/Projects/AMI")

from mothAI.trapdata_prediction_scripts.localization_classification import (
    localization_classification,
)

MODEL_BASE_PATH = pathlib.Path(__file__).parent.parent / "models"


def detect_and_classify(source_dir):
    source_dir = pathlib.Path(source_dir)
    args = argparse.Namespace(
        data_dir=str(source_dir.parent)
        + os.sep,  # Add trailing slash or other OS-native separator
        image_folder=str(source_dir.name),
        model_localize=MODEL_BASE_PATH / "v1_localizmodel_2021-08-17-12-06.pt",
        model_moth=MODEL_BASE_PATH / "mothsv2_20220421_110638_30.pth",
        model_moth_nonmoth=MODEL_BASE_PATH
        / "moth-nonmoth-effv2b3_20220506_061527_30.pth",
        category_map_moth=MODEL_BASE_PATH / "03-mothsv2_category_map.json",
        category_map_moth_nonmoth=MODEL_BASE_PATH / "05-moth-nonmoth_category_map.json",
    )
    annotations_path = localization_classification(args)
    annotations_target_path = source_dir / "detections.json"
    shutil.copy(str(annotations_path), str(annotations_target_path))
    annotations = json.load(open(annotations_target_path))
    return annotations
