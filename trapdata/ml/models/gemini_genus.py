"""
Genus-level second opinion from a vision language model, served through OpenRouter.

This is not a replacement for the species classifier. Measured blind on held-out crops,
Gemini 3-Flash reaches ~82% at genus but only ~74% at species, while the BioCLIP 2.5
head reaches 94.8% top-1 over the full 749-class species problem. So the VLM is used the
only way the numbers justify: as an independent genus-level check on the species head,
re-ranking the genera of its own top predictions.

Set AMI_OPENROUTER_API_KEY. Set AMI_GEMINI_MODEL to override the model.
"""

import base64
import io
import json
import os
import typing
import urllib.request

from trapdata.common.logs import logger

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = (
    "You are an expert lepidopterist identifying moths from automated light-trap camera "
    "crops. Look at the moth in this image. Which ONE of these {rank} does it belong to?\n\n"
    "{choices}\n\n"
    "Consider wing shape, resting posture (wings folded over the body as in Tortricidae, "
    "versus spread flat as in Geometridae), wing pattern and proportions. "
    "Reply with ONLY the exact name from the list, nothing else."
)


class GeminiGenusChecker:
    """Re-ranks candidate genera for a single crop. Stateless, one HTTP call per crop."""

    model: str = os.environ.get("AMI_GEMINI_MODEL", "google/gemini-3-flash-preview")
    timeout: int = 120
    max_tokens: int = 3000
    # Upscale small crops; the trap crops are often under 120 px and the API resamples badly.
    target_px: int = 320

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("AMI_OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("AMI_OPENROUTER_API_KEY is not set")

    def _encode(self, image) -> str:
        factor = max(1, int(self.target_px / max(image.width, image.height)))
        if factor > 1:
            image = image.resize((image.width * factor, image.height * factor))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=92)
        return base64.b64encode(buffer.getvalue()).decode()

    def predict(self, image, candidates: list[str], rank: str = "genera") -> tuple[str | None, str]:
        """Return (chosen label or None, raw reply)."""
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": PROMPT.format(
                                    rank=rank,
                                    choices="\n".join(f"- {c}" for c in candidates),
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + self._encode(image)
                                },
                            },
                        ],
                    }
                ],
            }
        ).encode()
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            response = json.load(urllib.request.urlopen(request, timeout=self.timeout))
        except Exception as exc:
            logger.warning(f"Gemini genus check failed: {type(exc).__name__}: {exc}")
            return None, ""

        message = response["choices"][0]["message"]
        # Reasoning-capable models can put the answer in `reasoning` and leave content null.
        text = ((message.get("content") or "") + " " + (message.get("reasoning") or "")).strip()
        chosen = next((c for c in candidates if c.lower() in text.lower()), None)
        return chosen, text


def genus_of(species_name: str) -> str:
    return species_name.split()[0]


def candidate_genera(labels: typing.Sequence[str], scores: typing.Sequence[float], k: int = 5) -> list[str]:
    """
    The genera of the species head's top predictions, in confidence order.

    Keeping the list short matters: the 82% blind accuracy was measured on a ~5-way
    choice, and a several-hundred-way list is a different and much harder task.
    """
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    out: list[str] = []
    for i in ranked:
        g = genus_of(labels[i])
        if g not in out:
            out.append(g)
        if len(out) >= k:
            break
    return out


def candidate_species(labels, scores, k: int = 5) -> list[str]:
    """The species head's top-k species, in confidence order.

    Kept short for the same reason as candidate_genera: the measured VLM accuracy was on
    a ~5-way choice. A 749-way prompt is a different and far harder task.
    """
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [labels[i] for i in ranked[:k]]
