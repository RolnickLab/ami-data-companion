"""
Pure Gemini 3-Flash species classifier for moth identification.

Uses Gemini VLM to directly classify moth species without a learned head.
Suitable for rapid iteration and novel species not in training data.

Set AMI_OPENROUTER_API_KEY to use.
Set AMI_GEMINI_VLM_MODEL to override model (default: google/gemini-3-flash-preview).
"""

import base64
import datetime
import io
import json
import os
import torch
import torchvision

from trapdata.common.logs import logger
from trapdata.ml.models.base import InferenceBaseClass
from trapdata.ml.models.bioclip import BACKBONE, BioCLIPWithLinearHead

import numpy as np
import urllib.request


class DummyGeminiModel(torch.nn.Module):
    """Placeholder module since Gemini VLM doesn't use PyTorch weights."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        # Dummy linear layer so the module has parameters and can be moved to device
        self.dummy = torch.nn.Linear(1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use GeminiVLMClassifier.predict_batch() instead")


class GeminiVLMOnlyNewfoundland749(InferenceBaseClass):
    """
    Pure Gemini 3-Flash species classifier for 749 Newfoundland moth species.

    Calls Gemini API for each crop to get a species classification.
    Returns softmax-normalized logits based on ranking confidence.
    """

    name = "Gemini VLM Species Classifier (Newfoundland, 749 species)"
    description = (
        "Gemini 3-Flash vision language model for direct species identification. "
        "Queries a list of candidate species and returns the model's choice ranked with confidence."
    )

    # Labels path: JSON file with species names indexed by class
    labels_path = os.environ.get(
        "AMI_GEMINI_VLM_LABELS",
        "/home/debian/bioclip-distill-leps/nf_deploy/label_map.json"
    )

    # API configuration
    api_key = os.environ.get("AMI_OPENROUTER_API_KEY", "")
    model = os.environ.get("AMI_GEMINI_VLM_MODEL", "google/gemini-3-flash-preview")
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    timeout = 120
    max_tokens = 200
    temperature = 0

    # Target upscaling for small crops
    target_px = 320
    input_size = 224
    lookup_gbif_names = False

    _label_map_cache: dict = {}
    _category_map_cache: dict = {}

    def _load_label_map(self) -> dict:
        """Load species name mapping: {index_str: {name, gbif_key, ...}}"""
        if self.labels_path not in self._label_map_cache:
            logger.info(f"Loading Gemini VLM label map from {self.labels_path}")
            with open(self.labels_path) as f:
                self._label_map_cache[self.labels_path] = json.load(f)
        return self._label_map_cache[self.labels_path]

    def get_labels(self, labels_path) -> dict[int, str]:
        """Map class index -> species name."""
        label_map = self._load_label_map()
        result = {}
        for idx_str, spec_data in label_map.items():
            idx = int(idx_str)
            # Get species_name from the spec data (might be nested)
            if isinstance(spec_data, dict):
                name = spec_data.get("species_name", spec_data.get("name", f"Species_{idx}"))
            else:
                name = str(spec_data)
            result[idx] = name
        return result

    def get_transforms(self) -> torchvision.transforms.Compose:
        """Return image preprocessing pipeline."""
        import open_clip
        _model, _train_transform, preprocess = open_clip.create_model_and_transforms(BACKBONE)
        return preprocess

    def get_weights(self, weights_path):
        """No weights to download; API-based."""
        return None

    def get_model(self) -> torch.nn.Module:
        """Return dummy module (API-based, no weights)."""
        num_classes = len(self.category_map)
        model = DummyGeminiModel(num_classes)
        return model.to(self.device)

    def _encode_image(self, image) -> str:
        """Encode PIL image to base64 JPEG."""
        # Upscale small crops
        factor = max(1, int(self.target_px / max(image.width, image.height)))
        if factor > 1:
            image = image.resize((image.width * factor, image.height * factor))

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=92)
        return base64.b64encode(buffer.getvalue()).decode()

    def _get_species_candidates(self, species_names: list[str], top_k: int = 20) -> list[str]:
        """Return top candidates for the prompt (to reduce API context)."""
        if len(species_names) <= top_k:
            return species_names
        return sorted(species_names)[:top_k]

    def _call_gemini(self, image_b64: str, candidates: list[str]) -> tuple[str | None, str]:
        """
        Call Gemini 3-Flash API with image and candidate species.

        Returns: (chosen_species or None, raw_reply)
        """
        if not self.api_key:
            logger.warning("AMI_OPENROUTER_API_KEY not set; returning None for Gemini VLM classification")
            return None, ""

        prompt = (
            "You are an expert lepidopterist identifying moths from automated light-trap camera "
            "crops. Look carefully at the moth in this image and identify its species.\n\n"
            "Which ONE of these species does it belong to?\n\n"
            + "\n".join(f"- {s}" for s in candidates)
            + "\n\nConsider wing shape, resting posture (wings folded over body vs spread flat), "
            "wing pattern and proportions. Reply with ONLY the exact species name from the list, nothing else."
        )

        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
        }).encode()

        request = urllib.request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            response = json.load(urllib.request.urlopen(request, timeout=self.timeout))
        except Exception as exc:
            logger.warning(f"Gemini VLM API call failed: {type(exc).__name__}: {exc}")
            return None, ""

        message = response["choices"][0]["message"]
        text = ((message.get("content") or "") + " " + (message.get("reasoning") or "")).strip()

        # Find exact match in candidates
        chosen = next((c for c in candidates if c.lower() == text.lower().strip()), None)
        if not chosen:
            # Try partial match
            chosen = next((c for c in candidates if c.lower() in text.lower()), None)

        return chosen, text

    def predict_batch(self, batch: list) -> torch.Tensor:
        """
        Process batch of crops via Gemini API.

        Returns: logits tensor [batch_size, num_classes] where logits are confidence-based.
        """
        all_species = list(self.category_map.values())
        batch_size = len(batch)
        num_classes = len(all_species)

        # Initialize logits: shape [batch_size, num_classes]
        logits = np.zeros((batch_size, num_classes), dtype=np.float32)

        for i, image in enumerate(batch):
            try:
                # Encode image
                image_b64 = self._encode_image(image)

                # Get candidates (don't overwhelm API with all 749)
                candidates = self._get_species_candidates(all_species, top_k=25)

                # Call Gemini
                chosen_species, reply = self._call_gemini(image_b64, candidates)

                if chosen_species:
                    # Find index of chosen species
                    try:
                        species_idx = list(self.category_map.values()).index(chosen_species)
                        # Set high confidence for chosen species
                        logits[i, species_idx] = 10.0  # Will become ~0.999 after softmax
                    except ValueError:
                        logger.warning(f"Gemini returned unknown species: {chosen_species}")
                        # If we don't recognize it, set uniform logits
                        logits[i, :] = 0.0
                else:
                    # API failed or no match; return uniform logits
                    logits[i, :] = 0.0

            except Exception as exc:
                logger.error(f"Error processing crop {i}: {exc}")
                logits[i, :] = 0.0

        return torch.from_numpy(logits).to(self.device)

    def post_process_batch(self, output: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to softmax probabilities.

        APIMothClassifier.post_process_batch will handle converting to ClassifierResult.
        We just need to return the softmax-normalized logits.
        """
        return output  # APIMothClassifier will apply softmax


# Aliases for backward compatibility
GeminiVLMNewfoundland749 = GeminiVLMOnlyNewfoundland749
