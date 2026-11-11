"""
Training rows from embeddings computed ahead of time, offline, from iNaturalist/GBIF images.

The backbone is frozen, so an image only ever needs embedding once per backbone. These
vectors are computed in bulk (scripts/precompute_offline_vectors.py) and stored as Parquet,
one directory per species, so a run that needs a few hundred species reads only those
files and never touches an image or a GPU.

Layout under the root directory (AMI_OFFLINE_VECTORS_DIR):

    <root>/<backbone slug>/
        _manifest.json             backbone key, vector dimension, how vectors were made
        _taxa.parquet              taxon_key, taxon_name, rows: one line per species
        taxon_key=<key>/part-*.parquet
                                   taxon_key, taxon_name, image_id, image_url, split,
                                   backbone_key, vector (fixed-size list of float16)

The backbone is part of the path because vectors from two backbones live in different
spaces: a head fitted on one is meaningless over the other, so they must never be mixed.
"""

import dataclasses
import json
import pathlib
import re
import typing

import numpy as np

from trapdata import logger
from trapdata.ml.training import TrainingRow

MANIFEST_FILENAME = "_manifest.json"
TAXA_INDEX_FILENAME = "_taxa.parquet"
VECTOR_COLUMN = "vector"
COLUMNS = (
    "taxon_key",
    "taxon_name",
    "image_id",
    "image_url",
    "split",
    "backbone_key",
    VECTOR_COLUMN,
)
SOURCE = "offline"


class OfflineVectorsUnavailable(Exception):
    """Raised when no offline vectors exist for the requested backbone."""


def backbone_slug(backbone_key: str) -> str:
    """A directory name for a backbone key such as 'hf-hub:imageomics/bioclip-2.5-vith14'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", backbone_key).strip("_")


def taxon_partition(taxon_key: typing.Any) -> str:
    return f"taxon_key={taxon_key}"


def normalise_name(name: str) -> str:
    """Match names regardless of case and stray whitespace, which Antenna lists do vary in."""
    return " ".join(str(name).split()).casefold()


@dataclasses.dataclass
class OfflineSelection:
    rows: list[TrainingRow]
    # Requested name -> rows found for it. Only species with at least one row.
    found: dict[str, int]
    missing: list[str]
    backbone_key: str
    dimensions: int

    def summary(self) -> dict[str, typing.Any]:
        splits: dict[str, int] = {}
        for row in self.rows:
            splits[row.split] = splits.get(row.split, 0) + 1
        return {
            "backbone_key": self.backbone_key,
            "dimensions": self.dimensions,
            "rows": len(self.rows),
            "splits": splits,
            "species_requested": len(self.found) + len(self.missing),
            "species_found": len(self.found),
            "counts": self.found,
            "missing_species": self.missing,
        }


class OfflineVectorSource:
    """Read precomputed vectors for a list of species from a local Parquet tree."""

    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root).expanduser()

    def backbone_dir(self, backbone_key: str) -> pathlib.Path:
        directory = self.root / backbone_slug(backbone_key)
        if not (directory / TAXA_INDEX_FILENAME).exists():
            raise OfflineVectorsUnavailable(
                f"No offline vectors for backbone '{backbone_key}' under {self.root}. "
                f"Expected {directory / TAXA_INDEX_FILENAME}."
            )
        return directory

    def manifest(self, backbone_key: str) -> dict[str, typing.Any]:
        path = self.backbone_dir(backbone_key) / MANIFEST_FILENAME
        return json.loads(path.read_text()) if path.exists() else {}

    def taxa(self, backbone_key: str) -> dict[str, tuple[typing.Any, str]]:
        """Normalised species name -> (taxon key, name as stored)."""
        import pyarrow.parquet as pq

        table = pq.read_table(
            self.backbone_dir(backbone_key) / TAXA_INDEX_FILENAME,
            columns=["taxon_key", "taxon_name"],
        )
        keys = table.column("taxon_key").to_pylist()
        names = table.column("taxon_name").to_pylist()
        return {normalise_name(n): (k, n) for k, n in zip(keys, names)}

    def _read_species(
        self, directory: pathlib.Path, taxon_key: typing.Any
    ) -> tuple[list, list, np.ndarray]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        parts = sorted((directory / taxon_partition(taxon_key)).glob("*.parquet"))
        if not parts:
            return [], [], np.empty((0, 0), dtype=np.float32)
        # Each file is read on its own, without partition discovery: the key is already a
        # column, and inferring it again from the directory name would clash with it.
        table = pa.concat_tables(
            pq.read_table(
                p, columns=["image_id", "split", VECTOR_COLUMN], partitioning=None
            )
            for p in parts
        )
        vectors = table.column(VECTOR_COLUMN).combine_chunks()
        width = vectors.type.list_size
        flat = vectors.flatten().to_numpy(zero_copy_only=False)
        matrix = np.asarray(flat, dtype=np.float32).reshape(-1, width)
        return (
            table.column("image_id").to_pylist(),
            table.column("split").to_pylist(),
            matrix,
        )

    def load(
        self,
        species: typing.Iterable[str],
        backbone_key: str,
        max_per_species: int | None = None,
        all_as_train: bool = False,
    ) -> OfflineSelection:
        """
        Training rows for the requested species, labelled with the name as requested.

        all_as_train puts every row on the train side. Used when Antenna rows are present:
        the held-out comparison must then be on the user's own data, which is what the head
        will actually see, not on iNaturalist photos.
        """
        directory = self.backbone_dir(backbone_key)
        index = self.taxa(backbone_key)
        manifest = self.manifest(backbone_key)

        rows: list[TrainingRow] = []
        found: dict[str, int] = {}
        missing: list[str] = []
        dimensions = int(manifest.get("dimensions") or 0)

        for name in dict.fromkeys(species):  # de-duplicated, order kept
            match = index.get(normalise_name(name))
            if match is None:
                missing.append(name)
                continue
            image_ids, splits, matrix = self._read_species(directory, match[0])
            if not len(image_ids):
                missing.append(name)
                continue
            dimensions = dimensions or matrix.shape[1]
            if matrix.shape[1] != dimensions:
                raise ValueError(
                    f"Offline vectors for '{name}' have {matrix.shape[1]} dimensions, "
                    f"expected {dimensions}."
                )
            # Defensive: the head expects L2-normalised features, as the serving path makes.
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0, 1, norms)

            picked = _select(image_ids, splits, max_per_species, all_as_train)
            for i, split in picked:
                rows.append(
                    TrainingRow(
                        detection_id=None,
                        label=name,
                        split=split,
                        features=matrix[i],
                        source=SOURCE,
                        image_id=str(image_ids[i]),
                    )
                )
            found[name] = len(picked)

        logger.info(
            f"Loaded {len(rows)} offline rows for {len(found)} species "
            f"({len(missing)} missing) from {directory}"
        )
        return OfflineSelection(
            rows=rows,
            found=found,
            missing=missing,
            backbone_key=backbone_key,
            dimensions=dimensions,
        )


def _select(
    image_ids: list,
    splits: list,
    max_per_species: int | None,
    all_as_train: bool,
) -> list[tuple[int, str]]:
    """
    Pick rows deterministically, so the same request always trains on the same images.

    Stored splits are 'train', 'val' or 'test'. 'val' joins 'test': this service has no
    early stopping, so a validation set would go unused. The cap applies per split.
    """
    by_split: dict[str, list[int]] = {}
    for i, split in enumerate(splits):
        split = "train" if all_as_train else ("test" if split == "val" else str(split))
        by_split.setdefault(split, []).append(i)
    picked: list[tuple[int, str]] = []
    for split, indices in sorted(by_split.items()):
        indices.sort(key=lambda i: str(image_ids[i]))
        if max_per_species:
            indices = indices[:max_per_species]
        picked.extend((i, split) for i in indices)
    return picked
