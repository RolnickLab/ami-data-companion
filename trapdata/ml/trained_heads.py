"""
Serve the heads this service has retrained, alongside the one it shipped with.

A retrained head is useless if nothing can select it. Each one saved to disk is offered as
its own pipeline, so Antenna sees it in /info and a person can pick it the same way they
pick any other. The head it was trained from stays exactly where it was: a retrain adds a
choice, it never replaces one.
"""

import json
import os
import pathlib
import typing

from trapdata import logger


def _default_heads_dir() -> str:
    """
    Where the training endpoint writes heads.

    Beside the downloaded model weights, following how this package already stores
    anything large, but in its own directory: a training run must never overwrite the
    head the service is currently serving.
    """
    import torch

    return str(pathlib.Path(torch.hub.get_dir()) / "trained_heads")


TRAINED_HEADS_DIR = os.environ.get("BIOCLIP_TRAINED_HEADS_DIR") or _default_heads_dir()

HEAD_SUFFIX = ".npz"
LABELS_SUFFIX = ".label_map.json"

# Prefix for the pipeline slug of a retrained head. Antenna keys algorithms by this
# string, so changing it orphans everything already registered.
RETRAINED_PREFIX = "bioclip-2-5-retrained"


def slug_for(name: str) -> str:
    """The pipeline slug a retrained head of this name is offered under."""
    return f"{RETRAINED_PREFIX}-{name}"


class TrainedHead(typing.NamedTuple):
    """One head this service has produced, as found on disk."""

    name: str
    directory: pathlib.Path
    metadata: dict

    @property
    def pipeline_slug(self) -> str:
        return slug_for(self.name)


def discover(directory: str | None = None) -> list[TrainedHead]:
    """Every retrained head on disk, newest first."""
    root = pathlib.Path(directory or TRAINED_HEADS_DIR)
    if not root.is_dir():
        return []

    heads = []
    for head_path in sorted(root.glob(f"*{HEAD_SUFFIX}"), reverse=True):
        name = head_path.name[: -len(HEAD_SUFFIX)]
        labels_path = root / f"{name}{LABELS_SUFFIX}"
        if not labels_path.exists():
            logger.warn(f"Skipping retrained head '{name}': no label map beside it")
            continue

        metadata = {}
        try:
            metadata = json.loads(labels_path.read_text())
        except json.JSONDecodeError:
            logger.warn(f"Retrained head '{name}' has an unreadable label map; ignoring its metadata")

        heads.append(TrainedHead(name=name, directory=root, metadata=metadata))
    return heads
