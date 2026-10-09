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


class TrainedHead(typing.NamedTuple):
    """One head this service has produced, as found on disk."""

    name: str
    directory: pathlib.Path
    metadata: dict

    @property
    def pipeline_slug(self) -> str:
        return f"{RETRAINED_PREFIX}-{self.name}"


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

        metadata_path = root / f"{name}.json"
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                logger.warn(
                    f"Retrained head '{name}' has unreadable metadata; ignoring it"
                )

        heads.append(TrainedHead(name=name, directory=root, metadata=metadata))
    return heads


def make_classifier_class(head: TrainedHead) -> type:
    """
    Build the classifier class that serves one retrained head.

    The head is the only thing that differs from the classifier it was trained from, so
    the class is that classifier with the head read from disk instead of the Hub.
    """
    from trapdata.api.models.classification import MothClassifierBioCLIP25Newfoundland

    return type(
        f"MothClassifierRetrained_{head.name.replace('-', '_')}",
        (MothClassifierBioCLIP25Newfoundland,),
        {
            "name": f"BioCLIP 2.5 + LogReg head (retrained {head.name})",
            "description": (
                "A classifier head retrained from verified identifications. "
                f"Trained on {head.metadata.get('rows', 'an unrecorded number of')} crops."
            ),
            "key": head.pipeline_slug,
            "head_local_dir": str(head.directory),
            "head_filename": f"{head.name}{HEAD_SUFFIX}",
            "categories_filename": f"{head.name}{LABELS_SUFFIX}",
        },
    )


def register(classifier_choices: dict, directory: str | None = None) -> list[str]:
    """
    Add every retrained head on disk to the pipeline choices, and say which were added.

    Called at startup and again after a training run, so a head becomes selectable without
    restarting the service.
    """
    added = []
    for head in discover(directory):
        if head.pipeline_slug in classifier_choices:
            continue
        classifier_choices[head.pipeline_slug] = make_classifier_class(head)
        added.append(head.pipeline_slug)

    if added:
        logger.info(f"Registered {len(added)} retrained head(s): {added}")
    return added
