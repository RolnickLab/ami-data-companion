"""
Embed iNaturalist/GBIF images with a frozen BioCLIP backbone into the offline-vector layout
that trapdata/ml/offline_vectors.py reads (see docs/offline-vectors-tier.md).

Two ways to say which images:

    # A manifest: CSV or Parquet with columns
    #   taxon_key, taxon_name, image_id, split, and image_path (local) and/or image_url
    python scripts/precompute_offline_vectors.py --manifest manifest.parquet --out /data/offline-vectors

    # A folder per species: <image-dir>/<species name>/<image>.jpg
    python scripts/precompute_offline_vectors.py --image-dir ./by-species --out /data/offline-vectors

Each run writes new part files and never rewrites old ones, so a large job can be split
into shards (--shard i --num-shards n) and run on several machines, or resumed. The species
index (_taxa.parquet) is rebuilt from the part files' metadata at the end of each run.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import typing
import uuid

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trapdata.ml.offline_vectors import (  # noqa: E402
    MANIFEST_FILENAME,
    TAXA_INDEX_FILENAME,
    VECTOR_COLUMN,
    backbone_slug,
    taxon_partition,
)

DEFAULT_BACKBONE = "hf-hub:imageomics/bioclip-2.5-vith14"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# An encoder takes a list of PIL images and returns an (n, d) float array. The real one
# wraps open_clip; tests pass a fake so they never load the 4 GB model.
Encoder = typing.Callable[[list], np.ndarray]


def hashed_split(group_id: str, val_frac: float = 0.10, test_frac: float = 0.10) -> str:
    """
    Same rule as the leps training pipeline (bioclip-distill-leps step1_global.gbif_split):
    hash the observation, not the photo, so photos of one observation never straddle splits.
    """
    r = int(hashlib.md5(f"obs:{group_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if r < val_frac:
        return "val"
    if r < val_frac + test_frac:
        return "test"
    return "train"


def manifest_from_folder(image_dir: pathlib.Path) -> list[dict]:
    rows = []
    for species_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
        name = " ".join(species_dir.name.replace("_", " ").split())
        for path in sorted(species_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image_id = f"{species_dir.name}/{path.stem}"
            rows.append(
                {
                    # No external taxon id in this mode: the name is the key.
                    "taxon_key": name.replace(" ", "_"),
                    "taxon_name": name,
                    "image_id": image_id,
                    "image_path": str(path),
                    "image_url": None,
                    "split": hashed_split(image_id),
                }
            )
    return rows


def manifest_from_file(path: pathlib.Path) -> list[dict]:
    import pandas as pd

    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for column in ("taxon_key", "taxon_name", "image_id"):
        if column not in frame:
            raise SystemExit(f"Manifest {path} has no '{column}' column.")
    if "image_path" not in frame and "image_url" not in frame:
        raise SystemExit("Manifest needs an image_path or image_url column.")
    if "split" not in frame:
        frame["split"] = [hashed_split(str(i)) for i in frame["image_id"]]
    for column in ("image_path", "image_url"):
        if column not in frame:
            frame[column] = None
    return frame.to_dict("records")


def open_image(row: dict):
    import io

    import PIL.Image
    import requests

    if row.get("image_path"):
        image = PIL.Image.open(row["image_path"])
    else:
        response = requests.get(row["image_url"], timeout=60)
        response.raise_for_status()
        image = PIL.Image.open(io.BytesIO(response.content))
    return image.convert("RGB")


def open_clip_encoder(backbone_key: str, device: str) -> Encoder:
    """The frozen image tower, normalised exactly as the serving path normalises it."""
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(backbone_key)
    model.eval().to(device)

    def encode(images: list) -> np.ndarray:
        batch = torch.stack([preprocess(im) for im in images]).to(device)
        with torch.inference_mode():
            features = model.encode_image(batch).float()
        return features.cpu().numpy()

    return encode


def embed(
    rows: list[dict], encoder: Encoder, batch_size: int = 64
) -> tuple[list[dict], np.ndarray]:
    """Embed every readable image; unreadable ones are skipped and reported, not fatal."""
    kept: list[dict] = []
    chunks: list[np.ndarray] = []
    skipped = 0
    for start in range(0, len(rows), batch_size):
        batch_rows, images = [], []
        for row in rows[start : start + batch_size]:
            try:
                images.append(open_image(row))
                batch_rows.append(row)
            except Exception as e:
                skipped += 1
                print(f"skip {row.get('image_id')}: {e}", file=sys.stderr)
        if not images:
            continue
        vectors = np.asarray(encoder(images), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        chunks.append(vectors / np.where(norms == 0, 1, norms))
        kept.extend(batch_rows)
    if skipped:
        print(f"Skipped {skipped} unreadable image(s)", file=sys.stderr)
    if not chunks:
        return [], np.empty((0, 0), dtype=np.float32)
    return kept, np.concatenate(chunks)


def write_parts(
    rows: list[dict],
    vectors: np.ndarray,
    out_root: pathlib.Path,
    backbone_key: str,
    run_id: str,
) -> pathlib.Path:
    """One new part file per species under <out>/<backbone>/taxon_key=<key>/."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = out_root / backbone_slug(backbone_key)
    directory.mkdir(parents=True, exist_ok=True)
    dimensions = int(vectors.shape[1])

    by_taxon: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_taxon.setdefault(str(row["taxon_key"]), []).append(i)

    vector_type = pa.list_(pa.float16(), dimensions)
    for taxon_key, indices in by_taxon.items():
        flat = pa.array(vectors[indices].astype(np.float16).reshape(-1), pa.float16())
        table = pa.table(
            {
                "taxon_key": pa.array([taxon_key] * len(indices), pa.string()),
                "taxon_name": [str(rows[i]["taxon_name"]) for i in indices],
                "image_id": [str(rows[i]["image_id"]) for i in indices],
                "image_url": pa.array(
                    [rows[i].get("image_url") or None for i in indices], pa.string()
                ),
                "split": [str(rows[i]["split"]) for i in indices],
                "backbone_key": [backbone_key] * len(indices),
                VECTOR_COLUMN: pa.FixedSizeListArray.from_arrays(flat, dimensions),
            }
        )
        assert table.schema.field(VECTOR_COLUMN).type == vector_type
        partition = directory / taxon_partition(taxon_key)
        partition.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, partition / f"part-{run_id}.parquet", compression="zstd")

    manifest_path = directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("dimensions") and manifest["dimensions"] != dimensions:
        raise SystemExit(
            f"{directory} holds {manifest['dimensions']}-d vectors; this run made {dimensions}-d."
        )
    manifest.update(
        {
            "backbone_key": backbone_key,
            "dimensions": dimensions,
            "dtype": "float16",
            "l2_normalised": True,
            "updated_at": datetime.datetime.now().isoformat(),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return directory


def rebuild_index(directory: pathlib.Path) -> int:
    """Rewrite _taxa.parquet from the part files. Reads only footers and one name each."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys, names, counts = [], [], []
    for partition in sorted(directory.glob("taxon_key=*")):
        parts = sorted(partition.glob("*.parquet"))
        if not parts:
            continue
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
        first = pq.read_table(parts[0], columns=["taxon_name"], partitioning=None)
        keys.append(partition.name.split("=", 1)[1])
        names.append(first.column("taxon_name")[0].as_py())
        counts.append(rows)
    pq.write_table(
        pa.table({"taxon_key": keys, "taxon_name": names, "rows": counts}),
        directory / TAXA_INDEX_FILENAME,
    )
    return len(keys)


def run(
    rows: list[dict],
    out_root: pathlib.Path,
    backbone_key: str,
    encoder: Encoder,
    batch_size: int = 64,
    shard: int = 0,
    num_shards: int = 1,
) -> pathlib.Path:
    rows = [r for i, r in enumerate(rows) if i % num_shards == shard]
    kept, vectors = embed(rows, encoder, batch_size=batch_size)
    if not kept:
        raise SystemExit("No image could be embedded.")
    run_id = f"{datetime.datetime.now():%Y%m%d%H%M%S}-{shard}-{uuid.uuid4().hex[:6]}"
    directory = write_parts(kept, vectors, out_root, backbone_key, run_id)
    species = rebuild_index(directory)
    print(f"Wrote {len(kept)} vectors; {species} species indexed under {directory}")
    return directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=pathlib.Path)
    source.add_argument("--image-dir", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)

    rows = (
        manifest_from_file(args.manifest)
        if args.manifest
        else manifest_from_folder(args.image_dir)
    )
    run(
        rows,
        args.out,
        args.backbone,
        open_clip_encoder(args.backbone, args.device),
        batch_size=args.batch_size,
        shard=args.shard,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
