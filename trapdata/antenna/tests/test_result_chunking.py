"""Tests for byte-bounded chunking of result POSTs to the Antenna API.

These tests reproduce the production failure mode where a single result POST for
a wide-taxonomy pipeline (e.g. the global moths model, ~29k classes) grew to
100+ MB and was rejected by the reverse proxy with HTTP 413. The fix splits the
results for one processed batch across multiple POSTs so each request body stays
under a configurable size cap.
"""

import datetime
import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

from trapdata.antenna.client import chunk_results_by_size, post_batch_results
from trapdata.antenna.schemas import AntennaTaskResult, AntennaTaskResults
from trapdata.api.schemas import (
    AlgorithmReference,
    BoundingBox,
    ClassificationResponse,
    DetectionResponse,
    PipelineResultsResponse,
    SourceImageResponse,
)


def _make_detection(image_id: str, num_classes: int) -> DetectionResponse:
    """Build a DetectionResponse carrying full-width classification arrays.

    This mirrors what the global moths classifier emits: one classification with
    labels/scores/logits arrays each of length ``num_classes``.
    """
    labels = [f"Genus_species_{i:06d}" for i in range(num_classes)]
    scores = [1.0 / num_classes] * num_classes
    logits = [float(i) for i in range(num_classes)]
    return DetectionResponse(
        source_image_id=image_id,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        algorithm=AlgorithmReference(name="Object Detector", key="detector"),
        timestamp=datetime.datetime(2024, 1, 1, 0, 0, 0),
        classifications=[
            ClassificationResponse(
                classification=labels[0],
                labels=labels,
                scores=scores,
                logits=logits,
                algorithm=AlgorithmReference(
                    name="Global Species Classifier", key="global_moths_2024"
                ),
                timestamp=datetime.datetime(2024, 1, 1, 0, 0, 0),
            )
        ],
    )


def _make_result(
    image_id: str, num_classes: int, detections_per_image: int = 1
) -> AntennaTaskResult:
    """Build one AntennaTaskResult for a single image with N wide detections."""
    detections = [
        _make_detection(image_id, num_classes) for _ in range(detections_per_image)
    ]
    return AntennaTaskResult(
        reply_subject=f"reply.{image_id}",
        result=PipelineResultsResponse(
            pipeline="global_moths_2024",
            source_images=[
                SourceImageResponse(id=image_id, url=f"http://x/{image_id}")
            ],
            detections=detections,
            total_time=1.0,
        ),
    )


def _serialized_body_size(results: list[AntennaTaskResult]) -> int:
    """Size in bytes of the JSON body actually sent for a list of results."""
    payload = AntennaTaskResults(results=results).model_dump(mode="json")
    return len(json.dumps(payload).encode("utf-8"))


class TestChunkResultsBySize(TestCase):
    """Unit tests for the greedy byte-bounded packer."""

    def test_empty_results_yields_no_chunks(self):
        self.assertEqual(chunk_results_by_size([], max_bytes=1000), [])

    def test_each_chunk_stays_under_cap(self):
        # ~29k-class detections: each result is ~2 MB. A cap of 5 MB should pack
        # at most ~2 results per chunk.
        num_classes = 29_000
        results = [_make_result(f"img{i}", num_classes) for i in range(8)]
        results_json = AntennaTaskResults(results=results).model_dump(mode="json")[
            "results"
        ]

        max_bytes = 5 * 1024 * 1024
        chunks = chunk_results_by_size(results_json, max_bytes=max_bytes)

        self.assertGreater(len(chunks), 1, "expected the batch to be split")
        for chunk in chunks:
            body_size = len(json.dumps({"results": chunk}).encode("utf-8"))
            self.assertLessEqual(
                body_size,
                max_bytes,
                f"chunk body {body_size} exceeded cap {max_bytes}",
            )

    def test_no_results_dropped(self):
        num_classes = 1_000
        results = [_make_result(f"img{i}", num_classes) for i in range(10)]
        results_json = AntennaTaskResults(results=results).model_dump(mode="json")[
            "results"
        ]
        chunks = chunk_results_by_size(results_json, max_bytes=500_000)
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, len(results))

    def test_oversize_single_result_gets_own_chunk(self):
        # One result larger than the cap cannot be split below one image; it must
        # still be emitted (in its own chunk) rather than dropped.
        num_classes = 29_000
        results = [_make_result("big", num_classes, detections_per_image=4)]
        results_json = AntennaTaskResults(results=results).model_dump(mode="json")[
            "results"
        ]
        chunks = chunk_results_by_size(results_json, max_bytes=1 * 1024 * 1024)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 1)


class TestPostBatchResultsChunking(TestCase):
    """post_batch_results must split a large batch across multiple POST bodies."""

    def test_large_batch_split_into_multiple_under_cap_posts(self):
        num_classes = 29_000
        # 8 images, each a single wide detection (~2 MB) -> ~15 MB total.
        results = [_make_result(f"img{i}", num_classes) for i in range(8)]

        max_bytes = 4 * 1024 * 1024
        posted_bodies: list[int] = []

        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {
            "status": "accepted",
            "job_id": 1,
            "results_queued": 0,
        }

        def fake_post(url, json=None, timeout=None):
            body_size = len(__import__("json").dumps(json).encode("utf-8"))
            posted_bodies.append(body_size)
            return fake_response

        fake_session = MagicMock()
        fake_session.post.side_effect = fake_post
        fake_session.__enter__.return_value = fake_session
        fake_session.__exit__.return_value = False

        with patch(
            "trapdata.antenna.client.get_http_session", return_value=fake_session
        ):
            ok = post_batch_results(
                base_url="http://x/api/v2",
                auth_token="t",
                job_id=1,
                results=results,
                max_bytes=max_bytes,
            )

        self.assertTrue(ok)
        self.assertGreater(len(posted_bodies), 1, "expected multiple POSTs")
        for size in posted_bodies:
            self.assertLessEqual(size, max_bytes, f"POST body {size} exceeded cap")

    def test_unsplit_baseline_would_exceed_cap(self):
        # Sanity check that the un-chunked body really is over the cap, so the
        # test above is exercising a real split, not a no-op.
        num_classes = 29_000
        results = [_make_result(f"img{i}", num_classes) for i in range(8)]
        body_size = _serialized_body_size(results)
        self.assertGreater(body_size, 4 * 1024 * 1024)

    def test_empty_results_is_noop_success(self):
        with patch("trapdata.antenna.client.get_http_session") as get_session:
            ok = post_batch_results(
                base_url="http://x/api/v2",
                auth_token="t",
                job_id=1,
                results=[],
                max_bytes=1000,
            )
        self.assertTrue(ok)
        get_session.assert_not_called()
