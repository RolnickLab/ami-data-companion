"""
Build the category map for a classifier trained with mini_trainer, such as Nemo
(MAMBO v3), from its checkpoint.

mini_trainer checkpoints label their classes with GBIF taxon IDs. Antenna matches
classifier labels to taxa by name, so this writes the usual {name: index} label map,
with each name taken from GBIF's canonicalName, as fetch_gbif_species does. The
indexes are the model's output indexes, read from the checkpoint.

The GBIF backbone lists some species twice under one name, once ACCEPTED and once
DOUBTFUL, and a model may have learned both. Only the ACCEPTED class is written; the
model drops classes that are not in its label map (see MiniTrainerClassifier).

Usage:
    python scripts/build_mini_trainer_category_map.py CHECKPOINT OUTPUT.json
"""

import argparse
import concurrent.futures
import json
import pathlib
import sys
import threading
import time

import requests

from trapdata.ml.utils import USER_AGENT, load_model_checkpoint

GBIF_SPECIES_URL = "https://api.gbif.org/v1/species/{gbif_id}"

# One kept-alive session per thread. fetch_gbif_species opens a new connection per
# lookup, about 2 lookups a second, too slow for a label map of 10,000+ species.
_sessions = threading.local()


def leaf_labels(checkpoint_path: pathlib.Path) -> list[str]:
    """Return the GBIF IDs of the model's leaf classes, in output order."""
    checkpoint = load_model_checkpoint(checkpoint_path, "cpu")
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    for key, value in state_dict.items():
        if key.endswith("._extra_state") and isinstance(value, dict):
            if "mini_trainer_version" in value:
                # cls2idx is keyed by rank, leaves ("0") first
                leaf_index = value["cls2idx"]["0"]
                labels = sorted(leaf_index, key=leaf_index.__getitem__)
                if [leaf_index[label] for label in labels] != list(range(len(labels))):
                    raise ValueError("Leaf class indexes are not contiguous")
                return labels
    raise ValueError(f"No mini_trainer metadata in {checkpoint_path}")


def fetch_name(gbif_id: str, attempts: int = 5) -> tuple[str, str] | None:
    """Return GBIF's canonical name and taxonomic status, or None if GBIF lacks it."""
    session = getattr(_sessions, "session", None)
    if session is None:
        session = _sessions.session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
    url = GBIF_SPECIES_URL.format(gbif_id=gbif_id)
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return data.get("canonicalName"), data.get("taxonomicStatus")
        except requests.exceptions.RequestException:
            # GBIF drops some connections under load
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    return None


def build_category_map(gbif_ids: list[str], max_workers: int) -> dict[str, int]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        taxa = list(executor.map(fetch_name, gbif_ids))

    missing = [
        gbif_id for gbif_id, taxon in zip(gbif_ids, taxa) if not taxon or not taxon[0]
    ]
    if missing:
        raise ValueError(f"No GBIF name for {len(missing)} IDs: {missing[:20]}")

    indexes_by_name: dict[str, list[int]] = {}
    for index, (name, _status) in enumerate(taxa):
        indexes_by_name.setdefault(name, []).append(index)

    category_map: dict[str, int] = {}
    for name, indexes in indexes_by_name.items():
        if len(indexes) > 1:
            accepted = [i for i in indexes if taxa[i][1] == "ACCEPTED"]
            if len(accepted) != 1:
                details = [(gbif_ids[i], taxa[i][1]) for i in indexes]
                raise ValueError(f"Cannot choose one class for {name!r}: {details}")
            dropped = [gbif_ids[i] for i in indexes if i != accepted[0]]
            print(
                f"{name}: keeping ACCEPTED GBIF {gbif_ids[accepted[0]]}, "
                f"dropping {dropped}",
                file=sys.stderr,
            )
            indexes = accepted
        category_map[name] = indexes[0]
    return category_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    gbif_ids = leaf_labels(args.checkpoint)
    category_map = build_category_map(gbif_ids, args.max_workers)
    args.output.write_text(
        json.dumps(category_map, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Wrote {len(category_map)} categories to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
