"""
Training from precomputed iNat/GBIF vectors (the "offline" tier), alone or on top of the
rows Antenna sends. Everything runs on a tiny synthetic Parquet tree: no model, no images
beyond a few generated pixels, no network.
"""

import asyncio
import importlib.util
import pathlib

import fastapi
import numpy as np
import pydantic
import pytest

from trapdata.ml import offline_vectors, training

DIM = 8
BACKBONE = "hf-hub:test/tiny-backbone"
SPECIES = ["Alpha one", "Beta two", "Gamma three"]

_script_path = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "precompute_offline_vectors.py"
)
_spec = importlib.util.spec_from_file_location(
    "precompute_offline_vectors", _script_path
)
precompute = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(precompute)


def _centre(index: int) -> np.ndarray:
    centre = np.zeros(DIM, dtype=np.float32)
    centre[index % DIM] = 1.0
    return centre


def _write_offline_tree(
    root: pathlib.Path, species=SPECIES, train=8, test=3, val=1
) -> pathlib.Path:
    """One direction per species, so a head can fit; written with the precompute writer."""
    rng = np.random.default_rng(1)
    rows, vectors = [], []
    for index, name in enumerate(species):
        for split, n in (("train", train), ("test", test), ("val", val)):
            for k in range(n):
                rows.append(
                    {
                        "taxon_key": str(1000 + index),
                        "taxon_name": name,
                        "image_id": f"{index}-{split}-{k}",
                        "image_url": f"https://inat.example/{index}/{split}/{k}.jpg",
                        "split": split,
                    }
                )
                vectors.append(_centre(index) + rng.normal(0, 0.01, DIM))
    directory = precompute.write_parts(
        rows, np.asarray(vectors, dtype=np.float32), root, BACKBONE, run_id="fixture"
    )
    precompute.rebuild_index(directory)
    return root


def _antenna_rows(species, per_split=(4, 3)):
    rng = np.random.default_rng(2)
    rows, detection_id = [], 0
    for index, name in enumerate(SPECIES):
        if name not in species:
            continue
        for split, n in zip(("train", "test"), per_split):
            for _ in range(n):
                detection_id += 1
                rows.append(
                    training.TrainingRow(
                        detection_id=detection_id,
                        label=name,
                        split=split,
                        features=(_centre(index) + rng.normal(0, 0.01, DIM)).astype(
                            np.float32
                        ),
                    )
                )
    return rows


# --- the loader -------------------------------------------------------------------


def test_the_loader_returns_rows_for_the_species_asked_for(tmp_path):
    _write_offline_tree(tmp_path)
    selection = offline_vectors.OfflineVectorSource(tmp_path).load(
        ["alpha  ONE", "Beta two"], backbone_key=BACKBONE
    )

    assert selection.missing == []
    # Labelled with the name as requested, so it lines up with the caller's species list.
    assert set(selection.found) == {"alpha  ONE", "Beta two"}
    assert {r.label for r in selection.rows} == {"alpha  ONE", "Beta two"}
    assert all(r.source == "offline" and r.detection_id is None for r in selection.rows)
    # 'val' folds into 'test': this service has no early stopping to use it for.
    splits = selection.summary()["splits"]
    assert splits == {"train": 16, "test": 8}
    assert selection.rows[0].features.shape == (DIM,)
    assert np.allclose(np.linalg.norm(selection.rows[0].features), 1.0, atol=1e-3)


def test_missing_species_are_reported_not_fatal(tmp_path):
    _write_offline_tree(tmp_path)
    selection = offline_vectors.OfflineVectorSource(tmp_path).load(
        ["Alpha one", "Nowhere to be found"], backbone_key=BACKBONE
    )
    assert selection.missing == ["Nowhere to be found"]
    assert list(selection.found) == ["Alpha one"]


def test_the_cap_is_per_species_and_split_and_deterministic(tmp_path):
    _write_offline_tree(tmp_path)
    source = offline_vectors.OfflineVectorSource(tmp_path)
    first = source.load(["Alpha one"], backbone_key=BACKBONE, max_per_species=2)
    second = source.load(["Alpha one"], backbone_key=BACKBONE, max_per_species=2)
    assert first.summary()["splits"] == {"train": 2, "test": 2}
    assert [r.image_id for r in first.rows] == [r.image_id for r in second.rows]


def test_with_antenna_rows_every_offline_row_trains(tmp_path):
    _write_offline_tree(tmp_path)
    selection = offline_vectors.OfflineVectorSource(tmp_path).load(
        ["Alpha one"], backbone_key=BACKBONE, all_as_train=True
    )
    assert selection.summary()["splits"] == {"train": 12}


def test_another_backbone_is_refused_rather_than_mixed(tmp_path):
    _write_offline_tree(tmp_path)
    with pytest.raises(offline_vectors.OfflineVectorsUnavailable):
        offline_vectors.OfflineVectorSource(tmp_path).load(
            ["Alpha one"], backbone_key="hf-hub:someone/else"
        )


# --- the precompute script --------------------------------------------------------


def test_the_precompute_script_writes_what_the_loader_reads(tmp_path):
    from PIL import Image

    images = tmp_path / "images"
    for index, name in enumerate(["Alpha_one", "Beta_two"]):
        folder = images / name
        folder.mkdir(parents=True)
        for k in range(5):
            Image.new("RGB", (4, 4), color=(index * 200, k * 40, 0)).save(
                folder / f"{k}.jpg"
            )
    (images / "Alpha_one" / "notes.txt").write_text("not an image")

    calls = []

    def fake_encoder(batch):
        # Stands in for the 4 GB backbone: a vector from the mean colour of each image.
        calls.append(len(batch))
        out = np.zeros((len(batch), DIM), dtype=np.float32)
        for i, im in enumerate(batch):
            out[i, :3] = np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) + 1
        return out

    rows = precompute.manifest_from_folder(images)
    assert len(rows) == 10
    out = tmp_path / "vectors"
    directory = precompute.run(rows, out, BACKBONE, fake_encoder, batch_size=4)

    assert calls == [4, 4, 2]
    assert directory == out / offline_vectors.backbone_slug(BACKBONE)
    manifest = offline_vectors.OfflineVectorSource(out).manifest(BACKBONE)
    assert manifest["dimensions"] == DIM and manifest["l2_normalised"] is True

    selection = offline_vectors.OfflineVectorSource(out).load(
        ["Alpha one", "Beta two"], backbone_key=BACKBONE
    )
    assert selection.missing == []
    assert selection.found == {"Alpha one": 5, "Beta two": 5}

    # A second run adds part files and the index counts both.
    precompute.run(rows, out, BACKBONE, fake_encoder, batch_size=4)
    again = offline_vectors.OfflineVectorSource(out).load(["Alpha one"], BACKBONE)
    assert again.found == {"Alpha one": 10}


def test_the_split_follows_the_observation_not_the_photo():
    assert precompute.hashed_split("obs-1") == precompute.hashed_split("obs-1")
    splits = {precompute.hashed_split(str(i)) for i in range(200)}
    assert splits == {"train", "val", "test"}


# --- POST /train ------------------------------------------------------------------


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    from trapdata.api import api

    _write_offline_tree(tmp_path / "offline")
    monkeypatch.setattr(api.settings, "offline_vectors_dir", tmp_path / "offline")
    monkeypatch.setattr(api, "_incumbent_head", lambda key: None)
    return api


def _fake_dataset(monkeypatch, rows, metadata=None):
    fetched = []

    def fetch(url):
        fetched.append(url)
        return rows, dict(metadata or {})

    monkeypatch.setattr(training, "fetch_dataset", fetch)
    return fetched


def _request(**fields):
    from trapdata.api.schemas import TrainRequest

    base = {"algorithm_key": "some-head", "save": False, "epochs": 60}
    base.update(fields)
    return TrainRequest(**base)


def _train(api, request):
    return asyncio.run(api.train(request))


def test_a_taxa_list_alone_trains_a_head(api_module, monkeypatch):
    fetched = _fake_dataset(monkeypatch, [])
    response = _train(
        api_module,
        _request(use_offline_inat=True, classes=SPECIES, backbone_key=BACKBONE),
    )

    assert fetched == []  # no Antenna data was asked for
    assert response.labels == sorted(SPECIES)
    assert response.offline["species_found"] == 3
    assert response.offline["missing_species"] == []
    # With no Antenna rows, evaluation uses the offline held-out split.
    assert response.rows["test"] == 3 * (3 + 1)
    assert response.candidate_metrics["top1"] > 0.9


def test_offline_rows_add_to_antenna_rows_but_evaluation_stays_on_antenna(
    api_module, monkeypatch
):
    antenna = _antenna_rows(["Alpha one", "Beta two"])
    _fake_dataset(monkeypatch, antenna, {"classes": SPECIES})
    response = _train(
        api_module,
        _request(
            dataset_url="http://antenna/media/set.npz",
            use_offline_inat=True,
            backbone_key=BACKBONE,
        ),
    )

    antenna_test = sum(r.split == "test" for r in antenna)
    offline_rows = response.offline["rows"]
    assert response.offline["splits"] == {"train": offline_rows}
    assert response.rows["test"] == antenna_test
    assert response.rows["train"] == len(antenna) - antenna_test + offline_rows
    # Gamma three had no verified crops, but the offline tier taught it.
    assert response.counts["Gamma three"] == 12
    assert response.labels == sorted(SPECIES)


def test_species_without_offline_vectors_are_reported(api_module, monkeypatch):
    _fake_dataset(monkeypatch, [])
    response = _train(
        api_module,
        _request(
            use_offline_inat=True,
            classes=SPECIES + ["Delta missing"],
            backbone_key=BACKBONE,
        ),
    )
    assert response.offline["missing_species"] == ["Delta missing"]
    assert any("Delta missing" in w for w in response.warnings)


def test_flag_off_keeps_the_old_behaviour(api_module, monkeypatch):
    antenna = _antenna_rows(SPECIES)
    fetched = _fake_dataset(monkeypatch, antenna, {"classes": SPECIES})
    request = _request(dataset_url="http://antenna/media/set.npz")

    assert request.use_offline_inat is False
    response = _train(api_module, request)

    assert fetched == ["http://antenna/media/set.npz"]
    assert response.offline is None
    assert response.rows["total"] == len(antenna)


def test_an_old_style_request_still_validates():
    request = _request(dataset_url="http://antenna/media/set.npz")
    assert request.classes is None and request.backbone_key is None


def test_a_request_with_no_data_at_all_is_refused():
    with pytest.raises(pydantic.ValidationError):
        _request()
    with pytest.raises(pydantic.ValidationError):
        # Offline alone needs the species list: there is nothing else to take it from.
        _request(use_offline_inat=True)


def test_offline_asked_for_but_not_configured_is_refused(api_module, monkeypatch):
    monkeypatch.setattr(api_module.settings, "offline_vectors_dir", None)
    _fake_dataset(monkeypatch, [])
    with pytest.raises(fastapi.HTTPException) as caught:
        _train(
            api_module,
            _request(use_offline_inat=True, classes=SPECIES, backbone_key=BACKBONE),
        )
    assert caught.value.status_code == 422


def test_vectors_of_a_different_width_are_refused(api_module, monkeypatch):
    wide = [
        training.TrainingRow(
            detection_id=i, label="Alpha one", split=s, features=np.ones(DIM * 2)
        )
        for i, s in enumerate(["train", "train", "test"])
    ]
    _fake_dataset(monkeypatch, wide, {"classes": ["Alpha one"]})
    with pytest.raises(fastapi.HTTPException) as caught:
        _train(
            api_module,
            _request(
                dataset_url="http://antenna/media/set.npz",
                use_offline_inat=True,
                backbone_key=BACKBONE,
            ),
        )
    assert caught.value.status_code == 422
    assert "dimensions" in caught.value.detail
