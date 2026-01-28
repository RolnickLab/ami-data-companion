"""Shared test utilities for API tests."""

from contextlib import contextmanager
from pathlib import Path
from typing import Type
from unittest.mock import patch

from fastapi.testclient import TestClient

from trapdata.api.api import CLASSIFIER_CHOICES, APIMothClassifier
from trapdata.api.schemas import SourceImageRequest
from trapdata.api.tests.image_server import StaticFileTestServer


def get_test_image_urls(
    file_server: StaticFileTestServer,
    test_images_dir: Path,
    subdir: str = "vermont",
    num: int = 2,
) -> list[str]:
    """Get list of test image URLs from file server.

    Args:
        file_server: StaticFileTestServer instance
        test_images_dir: Base directory containing test images
        subdir: Subdirectory within test_images_dir (default: "vermont")
        num: Number of images to return (default: 2)

    Returns:
        List of image URLs from the file server
    """
    images_dir = test_images_dir / subdir
    source_image_urls = [
        file_server.get_url(f.relative_to(test_images_dir))
        for f in images_dir.glob("*.jpg")
    ][:num]
    return source_image_urls


def get_test_images(
    file_server: StaticFileTestServer,
    test_images_dir: Path,
    subdir: str = "vermont",
    num: int = 2,
) -> list[SourceImageRequest]:
    """Get list of SourceImageRequest objects for testing.

    Args:
        file_server: StaticFileTestServer instance
        test_images_dir: Base directory containing test images
        subdir: Subdirectory within test_images_dir (default: "vermont")
        num: Number of images to return (default: 2)

    Returns:
        List of SourceImageRequest objects with IDs and URLs
    """
    urls = get_test_image_urls(file_server, test_images_dir, subdir, num)
    source_images = [
        SourceImageRequest(id=str(i), url=url) for i, url in enumerate(urls)
    ]
    return source_images


def get_pipeline_class(
    slug: str = "quebec_vermont_moths_2023",
) -> Type[APIMothClassifier]:
    """Get classifier class by pipeline slug.

    Args:
        slug: Pipeline slug (default: "quebec_vermont_moths_2023")

    Returns:
        APIMothClassifier class for the specified pipeline
    """
    return CLASSIFIER_CHOICES[slug]


@contextmanager
def patch_antenna_api_requests(test_client: TestClient):
    """Patch requests.get/post to route through TestClient.

    This allows tests to mock the Antenna API by routing requests through
    a TestClient instead of making real HTTP calls. Only requests to
    http://testserver are mocked - other requests pass through normally.

    Args:
        test_client: FastAPI TestClient to route requests through

    Usage:
        with patch_antenna_api_requests(antenna_client):
            # Code that makes requests to Antenna API
            response = requests.get("http://testserver/api/v2/jobs")
    """
    import requests

    # Save original methods BEFORE patching
    original_session_get = requests.Session.get
    original_session_post = requests.Session.post

    def mock_session_get(self, url, **kwargs):
        """Mock Session.get - route testserver through TestClient, others pass through."""
        if "testserver" in url:
            path = url.replace("http://testserver", "")
            headers = kwargs.get("headers", {})
            params = kwargs.get("params", {})
            return test_client.get(path, headers=headers, params=params)
        else:
            # Let real HTTP requests through (e.g., to file server)
            return original_session_get(self, url, **kwargs)

    def mock_session_post(self, url, **kwargs):
        """Mock Session.post - route testserver through TestClient, others pass through."""
        if "testserver" in url:
            path = url.replace("http://testserver", "")
            headers = kwargs.get("headers", {})
            json_data = kwargs.get("json")
            return test_client.post(path, headers=headers, json=json_data)
        else:
            return original_session_post(self, url, **kwargs)

    # Patch Session methods (used by get_http_session)
    with patch.object(requests.Session, "get", mock_session_get):
        with patch.object(requests.Session, "post", mock_session_post):
            yield
