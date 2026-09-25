"""
Retraining a classifier head from labels people verified in Antenna.

The backbone is frozen, so a head is a linear layer over stored embeddings. None of this
touches an image or a backbone, which is why it can be tested with plain vectors.
"""

import json
import pathlib

import numpy as np
import pytest

from trapdata.api.models.classification import MothClassifierBioCLIP25Newfoundland
from trapdata.ml import trained_heads, training

EMBED_DIM = 8


def _rows(labels: list[str], per_label: int = 6, split: str = "train"):
    """Linearly separable rows, one direction per label, so a head can actually fit."""
    rng = np.random.default_rng(0)
    rows = []
    detection_id = 0
    for index, label in enumerate(labels):
        centre = np.zeros(EMBED_DIM, dtype=np.float32)
        centre[index % EMBED_DIM] = 1.0
        for _ in range(per_label):
            detection_id += 1
            vector = centre + rng.normal(0, 0.01, EMBED_DIM).astype(np.float32)
            rows.append(
                training.TrainingRow(
                    detection_id=detection_id,
                    label=label,
                    split=split,
                    features=vector,
                )
            )
    return rows


def test_a_head_learns_the_classes_it_was_given():
    rows = _rows(["Species one", "Species two"]) + _rows(
        ["Species one", "Species two"], per_label=4, split="test"
    )

    result = training.retrain(rows=rows, incumbent=None, min_per_species=2, epochs=50)

    assert result["labels"] == ["Species one", "Species two"]
    assert result["candidate_metrics"]["top1"] > 0.9


def test_the_declared_species_list_wins_over_the_rows():
    """
    Antenna sends the project's taxa list inside the dataset. A species with no verified
    crops yet must still be a class the head can predict, or the pipeline goes blind to it.
    """
    rows = _rows(["Species one", "Species two"]) + _rows(
        ["Species one", "Species two"], per_label=4, split="test"
    )
    declared = ["Species one", "Species two", "Never seen yet"]

    result = training.retrain(
        rows=rows,
        incumbent=None,
        min_per_species=2,
        epochs=50,
        declared_classes=declared,
    )

    assert result["labels"] == sorted(declared)


def test_a_head_shape_this_service_cannot_fit_is_refused():
    """Refused rather than quietly served as a linear head, which would be a silent lie."""
    rows = _rows(["Species one", "Species two"])

    with pytest.raises(training.UnsupportedHeadType):
        training.retrain(rows=rows, incumbent=None, head_type="mlp1")


def test_a_saved_head_can_be_found_and_served(tmp_path: pathlib.Path):
    """A retrained head is useless unless something can select it afterwards."""
    rows = _rows(["Species one", "Species two"]) + _rows(
        ["Species one", "Species two"], per_label=4, split="test"
    )
    result = training.retrain(rows=rows, incumbent=None, min_per_species=2, epochs=50)
    training.save_head(result, tmp_path, "run-1")

    found = trained_heads.discover(str(tmp_path))
    assert [head.name for head in found] == ["run-1"]

    choices: dict = {}
    added = trained_heads.register(choices, str(tmp_path))

    assert added == [found[0].pipeline_slug]
    classifier = choices[found[0].pipeline_slug]
    assert issubclass(classifier, MothClassifierBioCLIP25Newfoundland)
    # It reads the head it was trained into, not the one it was trained from.
    assert classifier.head_local_dir == str(tmp_path)
    assert classifier.trainable is True


def test_a_saved_head_writes_the_weights_and_labels_it_promises(tmp_path: pathlib.Path):
    rows = _rows(["Species one", "Species two"]) + _rows(
        ["Species one", "Species two"], per_label=4, split="test"
    )
    result = training.retrain(rows=rows, incumbent=None, min_per_species=2, epochs=50)

    saved = training.save_head(result, tmp_path, "run-1")

    checkpoint = np.load(saved["head"])
    assert checkpoint["W"].shape == (2, EMBED_DIM)
    assert checkpoint["b"].shape == (2,)
    with open(saved["labels"]) as f:
        written = json.load(f)
    assert written["labels"] == ["Species one", "Species two"]


def test_a_retrained_head_can_read_back_its_own_label_map(tmp_path: pathlib.Path):
    """
    The label file this service writes must be the one it can load.

    save_head stores the labels beside the run's counts and metrics, while a head
    published on the Hub is a plain index-to-name map. The loader has to accept both, or
    a head is trainable but not servable.
    """
    rows = _rows(["Species one", "Species two"]) + _rows(
        ["Species one", "Species two"], per_label=4, split="test"
    )
    result = training.retrain(rows=rows, incumbent=None, min_per_species=2, epochs=50)
    training.save_head(result, tmp_path, "run-1")

    choices: dict = {}
    trained_heads.register(choices, str(tmp_path))
    classifier = next(iter(choices.values()))

    # Reading the labels must not need the backbone.
    shim = classifier.__new__(classifier)
    assert shim.get_labels(None) == {0: "Species one", 1: "Species two"}


def test_a_hub_label_map_loads_as_plain_names():
    """
    The published head maps an index to a record, not to a string.

    Reading it as a string gives every class a dict for a name, which fails only later
    and confusingly, when warm-starting tries to look one up.
    """
    from trapdata.ml.models.bioclip import BioCLIPClassifier

    published = {
        "1": {"species_name": "Actias luna", "inat_taxon_id": 47916},
        "0": {"species_name": "Lymantria dispar", "inat_taxon_id": 47802},
    }

    assert BioCLIPClassifier.labels_from(published) == [
        "Lymantria dispar",
        "Actias luna",
    ]
    assert BioCLIPClassifier.labels_from({"0": "Plain name"}) == ["Plain name"]
    assert BioCLIPClassifier.labels_from({"labels": ["From a retrain"]}) == [
        "From a retrain"
    ]


def test_a_head_carrying_its_own_vocabulary_loads(tmp_path: pathlib.Path):
    """
    One published head keeps its label vocabulary inside the npz.

    The vocabulary is longer than the number of classes, because a species with no
    training data can never be predicted, so `classes` indexes into it rather than
    lining up with it. Reading them as parallel arrays silently mislabels every class.
    """
    from trapdata.ml.models.bioclip import BioCLIP25PanamaClassifier

    np.savez(
        tmp_path / "head_combined.npz",
        W=np.zeros((2, EMBED_DIM), dtype=np.float32),
        b=np.zeros(2, dtype=np.float32),
        # Two output classes drawn from a four-name vocabulary.
        classes=np.array([3, 1]),
        labels=np.array(["Unused one", "Second", "Unused two", "Fourth"], dtype=object),
    )

    classifier = type(
        "LocalPanama",
        (BioCLIP25PanamaClassifier,),
        {"head_local_dir": str(tmp_path)},
    )
    shim = classifier.__new__(classifier)

    assert shim.get_labels(None) == {0: "Fourth", 1: "Second"}
    _weight, _bias, labels = classifier.load_head_arrays()
    assert labels == ["Fourth", "Second"]


def _train_request(**overrides):
    from trapdata.api.schemas import TrainRequest

    fields = {
        "dataset_url": "http://antenna/media/training/set.npz",
        "algorithm_key": "some-head",
        "callback_token": "a-signed-token",
        "head_upload_url": "http://antenna/api/v2/jobs/7/training-head/",
    }
    fields.update(overrides)
    return TrainRequest(**fields)


def test_the_head_is_uploaded_so_it_outlives_this_service(tmp_path, monkeypatch):
    """
    The head is written to a cache directory, so a rebuild loses it.

    Uploading is what makes a version something a person can still promote later.
    """
    from trapdata.api import api

    head = tmp_path / "run.npz"
    head.write_bytes(b"the weights")
    labels = tmp_path / "run.label_map.json"
    labels.write_text('{"labels": []}')

    sent = {}

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "files": {
                    "head": {
                        "path": "algorithms/x/run.npz",
                        "url": "/media/algorithms/x/run.npz",
                    }
                }
            }

    def fake_post(url, files=None, headers=None, timeout=None):
        sent["url"] = url
        sent["headers"] = headers
        sent["names"] = sorted(files)
        sent["head_bytes"] = files["head"].read()
        return Reply()

    monkeypatch.setattr("requests.post", fake_post)

    url = api._upload_head(_train_request(), {"head": str(head), "labels": str(labels)})

    assert url == "/media/algorithms/x/run.npz"
    assert sent["url"] == "http://antenna/api/v2/jobs/7/training-head/"
    assert sent["names"] == ["head", "labels"]
    assert sent["head_bytes"] == b"the weights"
    # The same signed token that authorises the result callback.
    assert sent["headers"]["Authorization"] == "Token a-signed-token"


def test_a_failed_upload_does_not_fail_the_training(tmp_path, monkeypatch):
    """
    The run succeeded and the head is still on local disk.

    Raising here would tell the caller its training failed when it did not; it only lost
    the durable copy.
    """
    from trapdata.api import api

    head = tmp_path / "run.npz"
    head.write_bytes(b"the weights")

    def fake_post(*args, **kwargs):
        raise ConnectionError("Antenna is unreachable")

    monkeypatch.setattr("requests.post", fake_post)

    assert api._upload_head(_train_request(), {"head": str(head)}) is None
