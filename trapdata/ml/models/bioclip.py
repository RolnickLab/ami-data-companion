"""
BioCLIP 2.5 species classifiers for the Antenna worker.

A frozen BioCLIP 2.5 ViT-H/14 encoder with a linear logistic-regression head on top.
The head comes from an sklearn LogisticRegression fit on L2-normalised BioCLIP
embeddings, so a softmax over the linear output reproduces sklearn's multinomial
predict_proba exactly. Only the head is trained; the encoder is never fine-tuned.

The exported head is an npz holding W (n_classes x 1024), b and classes. `classes`
gives the label for each row of W, which keeps the head and the category map aligned
even when the class values are not 0..N-1.
"""

import json
import os

import torch
import torchvision

from trapdata.common.logs import logger

from .classification import SpeciesClassifier

BACKBONE = "hf-hub:imageomics/bioclip-2.5-vith14"


class BioCLIPWithLinearHead(torch.nn.Module):
    """Frozen BioCLIP encoder, L2-normalise, then a single Linear layer."""

    def __init__(self, encoder: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder.encode_image(images).float()
        # The head was fit on L2-normalised embeddings, so normalise here too.
        features = features / features.norm(dim=-1, keepdim=True)
        return self.head(features)


class BioCLIPLogRegClassifier(SpeciesClassifier):
    """
    Base for BioCLIP 2.5 + logistic-regression heads.

    Subclasses set `weights_path` to the head npz. `labels_path` is unused for heads
    that carry their vocabulary inside the npz.
    """

    backbone_name: str = BACKBONE
    input_size = 224
    lookup_gbif_names = False

    _head_cache: dict = {}

    def _head(self):
        """Load and cache (W, b, classes) for this head."""
        import numpy as np

        path = self.weights_path
        if path not in self._head_cache:
            logger.info(f"Loading BioCLIP head from {path}")
            checkpoint = np.load(path, allow_pickle=True)
            self._head_cache[path] = checkpoint
        return self._head_cache[path]

    def _label_for(self, source_class) -> str:
        raise NotImplementedError

    def get_weights(self, weights_path):
        # The head is a local file, not a URL to download.
        return weights_path

    def get_labels(self, labels_path) -> dict[int, str]:
        checkpoint = self._head()
        return {index: self._label_for(cls) for index, cls in enumerate(checkpoint["classes"])}

    def get_transforms(self) -> torchvision.transforms.Compose:
        import open_clip

        _model, _train_transform, preprocess = open_clip.create_model_and_transforms(self.backbone_name)
        return preprocess

    def get_model(self) -> torch.nn.Module:
        import open_clip

        checkpoint = self._head()
        weight, bias = checkpoint["W"], checkpoint["b"]

        encoder, _, _ = open_clip.create_model_and_transforms(self.backbone_name)
        embed_dim = encoder.visual.output_dim
        if weight.shape[1] != embed_dim:
            raise ValueError(
                f"Head was fit on {weight.shape[1]}-dim embeddings but {self.backbone_name} "
                f"produces {embed_dim}-dim embeddings."
            )

        head = torch.nn.Linear(embed_dim, weight.shape[0])
        head.weight.data = torch.from_numpy(weight).float()
        head.bias.data = torch.from_numpy(bias).float()

        model = BioCLIPWithLinearHead(encoder, head).to(self.device)
        model.eval()
        return model


class BioCLIPNewfoundland749(BioCLIPLogRegClassifier):
    """The deployed Newfoundland trap head. 749 species."""

    name = "BioCLIP 2.5 + LogReg head (Newfoundland, 749 species) - pull worker"
    description = (
        "Frozen BioCLIP 2.5 ViT-H/14 with a linear logistic-regression head trained on "
        "iNaturalist photos plus verified Newfoundland trap crops."
    )
    weights_path = os.environ.get(
        "AMI_BIOCLIP_NF_HEAD", "/mnt/melabbas/antenna-nf-species/head_749_session.npz"
    )
    labels_path = os.environ.get(
        "AMI_BIOCLIP_NF_LABELS", "/home/debian/bioclip-distill-leps/nf_deploy/label_map.json"
    )

    _label_map_cache: dict = {}

    def _label_map(self) -> dict:
        if self.labels_path not in self._label_map_cache:
            with open(self.labels_path) as f:
                self._label_map_cache[self.labels_path] = json.load(f)
        return self._label_map_cache[self.labels_path]

    def _label_for(self, source_class) -> str:
        return self._label_map()[str(int(source_class))]["species_name"]


class BioCLIPPanama900(BioCLIPLogRegClassifier):
    """The Panama (BCI + Mount Totumas) head. 900 trained classes."""

    name = "BioCLIP 2.5 + LogReg head (Panama, 900 species) - pull worker"
    description = (
        "Frozen BioCLIP 2.5 ViT-H/14 with a linear logistic-regression head trained on "
        "BCI and Mount Totumas trap crops plus iNaturalist coverage. "
        "5-fold CV: 96.2% top-1, 69.4% macro."
    )
    weights_path = os.environ.get(
        "AMI_BIOCLIP_PANAMA_HEAD", "/mnt/melabbas/bci-panama/head_combined.npz"
    )
    labels_path = None  # the vocabulary travels inside the npz

    def _label_for(self, source_class) -> str:
        # 1,095-name vocabulary, indexed by class value; only 900 of them are trained.
        return str(self._head()["labels"][int(source_class)])


class BioCLIPWithGeminiCheck:
    """
    Mixin: after the species head runs, ask a VLM for an independent opinion at some rank
    and attach it as an extra classification under its own algorithm reference.

    It is a companion algorithm, never the terminal determination, because the numbers
    only justify it as a check. Measured blind on held-out crops:

        genus    ~82%      species  ~74%      (Gemini 3-Flash, ~5-way choice)
        BioCLIP 2.5 head   94.8% top-1 over the full 749-class species problem

    So the VLM re-ranks the species head's own top-k rather than choosing freely, which is
    the setting those numbers were measured in. Enabled by AMI_OPENROUTER_API_KEY; a
    missing key or a failed call skips the check rather than failing the job.

    The ADC worker does its own batching and never calls Algorithm.run(), so this hangs
    off post_classification_hook, which the worker invokes after the terminal classifier.
    """

    check_rank: str = "genus"          # "genus" or "species"
    check_name: str = "Gemini 3-Flash genus check"
    check_key: str = "gemini-3-flash-genus-check"
    check_workers: int = 6
    check_top_k: int = 5

    def _candidates(self, labels, scores):
        from .gemini_genus import candidate_genera, candidate_species
        fn = candidate_species if self.check_rank == "species" else candidate_genera
        return fn(labels, scores, k=self.check_top_k)

    def _as_rank(self, species_name: str) -> str:
        from .gemini_genus import genus_of
        return species_name if self.check_rank == "species" else genus_of(species_name)

    def extra_algorithm_configs(self):
        """Declare the checker so Antenna will accept its classifications."""
        from trapdata.api.schemas import AlgorithmCategoryMapResponse, AlgorithmConfigResponse
        from .gemini_genus import genus_of

        values = sorted(self.category_map.values())
        if self.check_rank == "genus":
            values = sorted({genus_of(v) for v in values})
        rank = "GENUS" if self.check_rank == "genus" else "SPECIES"
        return [
            AlgorithmConfigResponse(
                name=self.check_name,
                key=self.check_key,
                task_type="classification",
                description=(
                    f"Independent {self.check_rank}-level second opinion from a vision "
                    "language model (Gemini 3-Flash via OpenRouter). Re-ranks the top "
                    f"{self.check_top_k} predictions of the species head. Measured blind on "
                    "held-out crops at ~82% for genus and ~74% for species, versus 94.8% "
                    "top-1 for the species head, so it is a check and not a determination."
                ),
                category_map=AlgorithmCategoryMapResponse(
                    data=[{"index": i, "label": v, "taxon_rank": rank} for i, v in enumerate(values)],
                    labels=values,
                    version="v1",
                    description=f"{len(values)} {self.check_rank} values from the species head's label set.",
                    uri=None,
                ),
                uri="https://openrouter.ai/google/gemini-3-flash-preview",
            )
        ]

    def _checker(self):
        if not hasattr(self, "_gemini"):
            from .gemini_genus import GeminiGenusChecker
            try:
                self._gemini = GeminiGenusChecker()
            except ValueError:
                logger.info("AMI_OPENROUTER_API_KEY not set; skipping the Gemini check")
                self._gemini = None
        return self._gemini

    def post_classification_hook(self, image_detections, image_tensors):
        checker = self._checker()
        if checker is None:
            return

        import concurrent.futures as cf
        import datetime
        import torchvision.transforms.functional as TF
        from trapdata.api.schemas import AlgorithmReference, ClassificationResponse

        labels = [self.category_map[i] for i in sorted(self.category_map)]
        rank_word = "species" if self.check_rank == "species" else "genera"

        jobs = []
        for image_id, detections in image_detections.items():
            tensor = image_tensors.get(image_id)
            if tensor is None:
                continue
            for detection in detections:
                terminal = [c for c in (detection.classifications or []) if c.terminal]
                if not terminal or not terminal[-1].scores:
                    continue
                top = terminal[-1]
                candidates = self._candidates(labels, top.scores)
                if len(candidates) < 2:
                    continue
                box = detection.bbox
                y1, y2, x1, x2 = int(box.y1), int(box.y2), int(box.x1), int(box.x2)
                if y1 >= y2 or x1 >= x2:
                    continue
                crop = TF.to_pil_image(tensor[:, y1:y2, x1:x2])
                jobs.append((detection, top, candidates, crop))

        if not jobs:
            return

        def ask(job):
            _detection, _top, candidates, crop = job
            chosen, _raw = checker.predict(crop, candidates, rank=rank_word)
            return job, chosen

        agreed = checked = 0
        with cf.ThreadPoolExecutor(max_workers=self.check_workers) as pool:
            for (detection, top, candidates, _crop), chosen in pool.map(ask, jobs):
                if not chosen:
                    continue
                checked += 1
                agreed += int(chosen == self._as_rank(top.classification))
                detection.classifications.append(
                    ClassificationResponse(
                        classification=chosen,
                        labels=candidates,
                        scores=[1.0 if c == chosen else 0.0 for c in candidates],
                        algorithm=AlgorithmReference(name=self.check_name, key=self.check_key),
                        timestamp=datetime.datetime.now(),
                        terminal=False,
                    )
                )
        if checked:
            logger.info(
                f"Gemini {self.check_rank} check: {agreed}/{checked} agreed with the "
                f"species head ({100 * agreed / checked:.0f}%)"
            )


class BioCLIPWithGeminiGenusCheck(BioCLIPWithGeminiCheck):
    check_rank = "genus"
    check_name = "Gemini 3-Flash genus check"
    check_key = "gemini-3-flash-genus-check"


class BioCLIPWithGeminiSpeciesCheck(BioCLIPWithGeminiCheck):
    check_rank = "species"
    check_name = "Gemini 3-Flash species check"
    check_key = "gemini-3-flash-species-check"
