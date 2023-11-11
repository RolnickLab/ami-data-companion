import pathlib
import typing
import unittest

import PIL.Image
import torch

from trapdata.common.filemanagement import find_images
from trapdata.ml.models.classification import (
    SimpleInferenceBaseClass,
    SimplePanamaClassifier,
    SimpleQuebecVermontClassifier,
    SimpleUKDenmarkClassifier,
)

TEST_IMAGES = pathlib.Path(__file__).parent / "images" / "cropped"
LOCAL_WEIGHTS_PATH = torch.hub.get_dir()


def load_cropped_images(
    subdir: str,
) -> typing.Generator[typing.Tuple[str, PIL.Image.Image], None, None]:
    for image in find_images(
        TEST_IMAGES / subdir, check_exif=False, skip_missing_timestamps=False
    ):
        path = image["path"]
        true_label = pathlib.Path(path).stem
        yield true_label, PIL.Image.open(path)


class TestSimpleClassifier(unittest.TestCase):
    def test_predict_singles(self):
        classifier = SimpleQuebecVermontClassifier(user_data_path=LOCAL_WEIGHTS_PATH)
        for true_name, image in load_cropped_images("vermont"):
            batch_results = classifier.predict([image])
            predicted_label, score = batch_results[0][0]
            self.assertEqual(true_name, predicted_label)

    def _test_predict_batch(
        self, Classifier: typing.Type[SimpleInferenceBaseClass], examples_dir: str
    ):
        classifier = Classifier(user_data_path=LOCAL_WEIGHTS_PATH)
        true_labels, images = list(zip(*load_cropped_images(examples_dir)))
        batch_results = classifier.predict(images)
        for true_label, predictions in zip(true_labels, batch_results):
            predicted_label, score = predictions[0]
            self.assertEqual(true_label, predicted_label)

    def test_quebec_vermont(self):
        self._test_predict_batch(SimpleQuebecVermontClassifier, "vermont")

    def test_panama(self):
        self._test_predict_batch(SimplePanamaClassifier, "panama")

    def test_uk_denmark(self):
        self._test_predict_batch(SimpleUKDenmarkClassifier, "denmark")


if __name__ == "__main__":
    unittest.main()
