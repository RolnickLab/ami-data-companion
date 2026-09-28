"""
Species classifiers built on a frozen BioCLIP backbone with a linear head on top.

The backbone never changes, so a crop's embedding never changes either. That is what
makes these classifiers retrainable: fitting a new head is a small matrix over stored
embeddings rather than another pass over the images. The forward pass therefore returns
the embedding alongside the logits, so whatever consumes a classification can keep the
vector that produced it.

The head is an sklearn LogisticRegression exported to a single Linear layer, so a softmax
over its output reproduces sklearn's multinomial predict_proba exactly. See
`scripts/export_logreg_head.py` for the conversion.
"""

import json
import os
import pathlib

import torch
import torchvision

from trapdata import logger

from .base import InferenceBaseClass
from .classification import SpeciesClassifier


class BioCLIPWithLinearHead(torch.nn.Module):
    """
    A frozen BioCLIP image encoder with a linear classification head on top.

    Encoder and head are kept in one module so the whole classifier loads and moves
    between devices as a single object.
    """

    def __init__(self, encoder: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, features)``. The features are what the head consumed."""
        features = self.encoder.encode_image(images).float()
        # The head was fit on L2-normalised embeddings, so normalise here too.
        features = features / features.norm(dim=-1, keepdim=True)
        return self.head(features), features


class BioCLIPClassifier(InferenceBaseClass):
    """
    Loads a BioCLIP backbone and a linear head from a Hugging Face repository.

    Subclasses name the backbone and the repository holding the head; everything else is
    shared. Unlike the other classifiers here, the weights and labels come from the Hub
    rather than the object store, so ``get_weights`` and ``get_labels`` are overridden.
    """

    # Retraining only has to fit the head, which is why these models are offered for it.
    trainable = True

    backbone_name: str = "hf-hub:imageomics/bioclip-2.5-vith14"
    head_repo_id: str = ""
    head_repo_type: str = "space"
    head_filename: str = "logreg_head_antenna.npz"
    categories_filename: str = "label_map.json"
    # Read the head from this directory instead of the Hub. A head this service retrained
    # only exists on disk, so that is the only way to serve one.
    head_local_dir: str | None = None

    def _head_file(self, filename: str) -> str:
        if self.head_local_dir:
            logger.info(f"Loading {filename} from {self.head_local_dir}")
            return str(pathlib.Path(self.head_local_dir) / filename)

        from huggingface_hub import hf_hub_download

        logger.info(
            f"Loading {filename} from {self.head_repo_id} ({self.head_repo_type})"
        )
        return hf_hub_download(
            repo_id=self.head_repo_id,
            filename=filename,
            repo_type=self.head_repo_type,
        )

    @staticmethod
    def labels_from(label_map) -> list[str]:
        """
        Normalise the three shapes a label map arrives in.

        A head published on the Hub maps an index to a record carrying the species name
        and its iNaturalist id; an older one maps an index straight to a name; a head
        retrained here stores its label list beside the counts and metrics of the run.
        All three have to load, or a head is trainable but not servable.
        """
        if isinstance(label_map, dict) and isinstance(label_map.get("labels"), list):
            return list(label_map["labels"])

        ordered = sorted(label_map.items(), key=lambda kv: int(kv[0]))
        return [
            entry["species_name"] if isinstance(entry, dict) else str(entry)
            for _, entry in ordered
        ]

    @classmethod
    def load_head_arrays(cls) -> tuple:
        """
        Read this classifier's head as ``(weight, bias, labels)`` without building it.

        Comparing a newly fitted head against the one in service only needs its numbers,
        and instantiating the classifier would pull the whole backbone into memory to
        answer a question about a single Linear layer.
        """
        import numpy as np

        # _head_file only reads class attributes, so it needs no initialised instance.
        shim = cls.__new__(cls)
        checkpoint = np.load(shim._head_file(cls.head_filename))
        with open(shim._head_file(cls.categories_filename)) as f:
            labels = cls.labels_from(json.load(f))
        return checkpoint["W"], checkpoint["b"], labels

    def get_weights(self, weights_path):
        """The head travels with its label map, so both are fetched together on load."""
        return self._head_file(self.head_filename)

    def get_labels(self, labels_path) -> dict[int, str]:
        """Read the label map that belongs to this head. See :meth:`labels_from`."""
        with open(self._head_file(self.categories_filename)) as f:
            return dict(enumerate(self.labels_from(json.load(f))))

    def get_model(self) -> torch.nn.Module:
        import numpy as np
        import open_clip

        encoder, _, preprocess = open_clip.create_model_and_transforms(
            self.backbone_name
        )
        self._preprocess = preprocess

        checkpoint = np.load(self.weights)
        weight, bias = checkpoint["W"], checkpoint["b"]
        embed_dim = encoder.visual.output_dim
        if weight.shape[1] != embed_dim:
            raise ValueError(
                f"Head was fit on {weight.shape[1]}-dim embeddings but "
                f"{self.backbone_name} produces {embed_dim}-dim embeddings."
            )

        head = torch.nn.Linear(embed_dim, weight.shape[0])
        head.weight.data = torch.from_numpy(weight).float()
        head.bias.data = torch.from_numpy(bias).float()

        model = BioCLIPWithLinearHead(encoder, head)
        model.to(self.device)
        model.eval()
        return model

    def get_transforms(self) -> torchvision.transforms.Compose:
        """
        BioCLIP ships the transform that matches its backbone, so use that rather than a
        hand-written one: a mismatched resize or normalisation silently degrades accuracy.
        """
        if not getattr(self, "_preprocess", None):
            import open_clip

            _, _, self._preprocess = open_clip.create_model_and_transforms(
                self.backbone_name
            )
        return self._preprocess


class BioCLIPVocabularyInHeadClassifier(BioCLIPClassifier):
    """
    A head whose label vocabulary travels inside the npz rather than beside it.

    The vocabulary is longer than the number of output classes — a species with no
    training data can never be predicted, so it is not a class — which is why the head's
    ``classes`` array indexes into it rather than lining up with it.
    """

    categories_filename = ""

    def get_labels(self, labels_path) -> dict[int, str]:
        import numpy as np

        checkpoint = np.load(self._head_file(self.head_filename), allow_pickle=True)
        vocabulary = checkpoint["labels"]
        return {
            index: str(vocabulary[int(source_class)])
            for index, source_class in enumerate(checkpoint["classes"])
        }

    @classmethod
    def load_head_arrays(cls) -> tuple:
        import numpy as np

        shim = cls.__new__(cls)
        checkpoint = np.load(shim._head_file(cls.head_filename), allow_pickle=True)
        vocabulary = checkpoint["labels"]
        labels = [
            str(vocabulary[int(source_class)]) for source_class in checkpoint["classes"]
        ]
        return checkpoint["W"], checkpoint["b"], labels


class BioCLIP25NewfoundlandClassifier(SpeciesClassifier, BioCLIPClassifier):
    name = "BioCLIP 2.5 + LogReg head (Newfoundland)"
    description = (
        "Frozen BioCLIP 2.5 ViT-H/14 with a logistic-regression head over the "
        "Newfoundland species list. The head can be retrained from verified crops."
    )
    head_repo_id = "mohammedelabbas/newfoundland-leps-trap-classifier"


class BioCLIP25PanamaClassifier(SpeciesClassifier, BioCLIPVocabularyInHeadClassifier):
    name = "BioCLIP 2.5 + LogReg head (Panama)"
    description = (
        "Frozen BioCLIP 2.5 ViT-H/14 with a logistic-regression head over the Panama "
        "(BCI and Mount Totumas) species list. The head can be retrained."
    )
    # Not published, so it is read from a directory this deployment provides.
    head_filename = "head_combined.npz"
    head_local_dir = os.environ.get("BIOCLIP_PANAMA_HEAD_DIR") or None
