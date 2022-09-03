import os
import json
import argparse
import pathlib
import shutil
import glob


import sys

# REPO_PATH = pathlib.Path(__file__).parent.parent

# @TODO Integrate classification into project
sys.path.append("/home/michael/Projects/AMI")
sys.path.append("/Users/michael/Projects/AMI")

from .localization_classification_batch import (
    localization_classification,
)

# @TODO make this a configurable setting
MODEL_BASE_PATH = pathlib.Path(__file__).parent.parent.parent / "data/models"


def detect_and_classify(base_directory, image_list, results_callback=None):
    base_directory = pathlib.Path(base_directory)
    args = argparse.Namespace(
        base_directory=base_directory,
        image_list=image_list,
        model_localize=MODEL_BASE_PATH / "v1_localizmodel_2021-08-17-12-06.pt",
        model_moth=MODEL_BASE_PATH / "mothsv2_20220421_110638_30.pth",
        model_moth_nonmoth=MODEL_BASE_PATH
        / "moth-nonmoth-effv2b3_20220506_061527_30.pth",
        category_map_moth=MODEL_BASE_PATH / "03-mothsv2_category_map.json",
        category_map_moth_nonmoth=MODEL_BASE_PATH / "05-moth-nonmoth_category_map.json",
    )

    localization_classification(args, results_callback=results_callback)


if __name__ == "__main__":
    source_dir = sys.argv[1]
    image_list = glob(source_dir)
    detect_and_classify(source_dir, image_list)
