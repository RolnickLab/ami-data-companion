"""Tests for rest_collate_fn in trapdata.antenna.datasets."""

import torch

from trapdata.antenna.datasets import rest_collate_fn


def _make_item(h=224, w=224, image_id="img1", reply_subject="reply1", error=None):
    """Helper to create a batch item dict."""
    item = {
        "image": None if error else torch.rand(3, h, w),
        "image_id": image_id,
        "reply_subject": reply_subject,
        "image_url": f"http://example.com/{image_id}.jpg",
    }
    if error:
        item["error"] = error
    return item


class TestRestCollateFnSameSize:
    """When all images are the same size, output should be a stacked tensor."""

    def test_single_image(self):
        batch = [_make_item(h=100, w=200)]
        result = rest_collate_fn(batch)
        assert isinstance(result["images"], torch.Tensor)
        assert result["images"].shape == (1, 3, 100, 200)

    def test_multiple_same_size(self):
        batch = [_make_item(h=100, w=200, image_id=f"img{i}") for i in range(4)]
        result = rest_collate_fn(batch)
        assert isinstance(result["images"], torch.Tensor)
        assert result["images"].shape == (4, 3, 100, 200)

    def test_metadata_preserved(self):
        batch = [
            _make_item(image_id="a", reply_subject="r1"),
            _make_item(image_id="b", reply_subject="r2"),
        ]
        result = rest_collate_fn(batch)
        assert result["image_ids"] == ["a", "b"]
        assert result["reply_subjects"] == ["r1", "r2"]
        assert len(result["image_urls"]) == 2


class TestRestCollateFnMixedSizes:
    """When images have different sizes, output should be a list of tensors."""

    def test_two_different_sizes(self):
        batch = [
            _make_item(h=2160, w=4096, image_id="big"),
            _make_item(h=2464, w=3280, image_id="small"),
        ]
        result = rest_collate_fn(batch)
        assert isinstance(result["images"], list)
        assert len(result["images"]) == 2
        assert result["images"][0].shape == (3, 2160, 4096)
        assert result["images"][1].shape == (3, 2464, 3280)

    def test_three_different_sizes(self):
        batch = [
            _make_item(h=100, w=200, image_id="a"),
            _make_item(h=300, w=400, image_id="b"),
            _make_item(h=500, w=600, image_id="c"),
        ]
        result = rest_collate_fn(batch)
        assert isinstance(result["images"], list)
        assert len(result["images"]) == 3

    def test_metadata_preserved(self):
        batch = [
            _make_item(h=100, w=200, image_id="a", reply_subject="r1"),
            _make_item(h=300, w=400, image_id="b", reply_subject="r2"),
        ]
        result = rest_collate_fn(batch)
        assert result["image_ids"] == ["a", "b"]
        assert result["reply_subjects"] == ["r1", "r2"]

    def test_warns_on_mixed_sizes(self, capsys):
        batch = [
            _make_item(h=100, w=200, image_id="a"),
            _make_item(h=300, w=400, image_id="b"),
        ]
        rest_collate_fn(batch)
        captured = capsys.readouterr()
        assert "different image sizes" in captured.out


class TestRestCollateFnFailedItems:
    """Failed items (image=None or error set) should be separated out."""

    def test_all_failed(self):
        batch = [
            _make_item(error="download failed", image_id="a"),
            _make_item(error="timeout", image_id="b"),
        ]
        result = rest_collate_fn(batch)
        assert "images" not in result
        assert result["reply_subjects"] == []
        assert result["image_ids"] == []
        assert len(result["failed_items"]) == 2

    def test_mixed_success_and_failure(self):
        batch = [
            _make_item(h=224, w=224, image_id="ok"),
            _make_item(error="bad url", image_id="fail"),
        ]
        result = rest_collate_fn(batch)
        assert isinstance(result["images"], torch.Tensor)
        assert result["images"].shape == (1, 3, 224, 224)
        assert result["image_ids"] == ["ok"]
        assert len(result["failed_items"]) == 1
        assert result["failed_items"][0]["image_id"] == "fail"

    def test_mixed_sizes_with_failure(self):
        batch = [
            _make_item(h=100, w=200, image_id="a"),
            _make_item(error="oops", image_id="b"),
            _make_item(h=300, w=400, image_id="c"),
        ]
        result = rest_collate_fn(batch)
        # Two successful images with different sizes → list
        assert isinstance(result["images"], list)
        assert len(result["images"]) == 2
        assert len(result["failed_items"]) == 1
