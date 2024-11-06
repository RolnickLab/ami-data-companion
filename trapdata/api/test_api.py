import base64
import os
import urllib.parse

import pytest
import requests
import responses
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from trapdata.api.api import PipelineChoice, PipelineRequest, SourceImageRequest, app
from trapdata.tests import TEST_IMAGES_BASE_PATH

app.mount("/images", StaticFiles(directory=TEST_IMAGES_BASE_PATH), name="images")


@pytest.fixture
def mock_image_request():
    with responses.RequestsMock() as rsps:

        def _mock_image(image_path):
            # Construct the full URL
            url = urllib.parse.urljoin(
                "http://testserver", app.url_path_for("images", path=image_path)
            )

            # Read the actual image file
            full_image_path = os.path.join(TEST_IMAGES_BASE_PATH, image_path)
            with open(full_image_path, "rb") as image_file:
                image_content = image_file.read()

            # Mock the response for the image URL
            rsps.add(
                responses.GET,
                url,
                body=image_content,
                status=200,
                content_type="image/jpeg",
            )

            return url

        yield _mock_image


@pytest.fixture
def client():
    return TestClient(app)


def get_test_image_path(random=False):
    return "vermont/20220622000459-108-snapshot.jpg"


def get_test_image_url():
    path = app.url_path_for("images", path=get_test_image_path())
    url = f"{client.base_url}{path}"
    print(f"Test image URL: {url}")
    return url


def get_test_image_base64():
    with open(get_test_image_path(), "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_test_pipeline_request(image_url: str):
    return PipelineRequest(
        pipeline=PipelineChoice["panama_moths_2024"],
        source_images=[SourceImageRequest(id="test_image_1", url=image_url)],
    )


def test_pipeline_request_format(client, mock_image_request):
    image_path = get_test_image_path()
    image_url = mock_image_request(image_path)

    requests.get(image_url)
    # Ensure the output json is correct
    pipeline_request = get_test_pipeline_request(image_url)
    expected_dict = {
        "pipeline": "panama_moths_2024",
        "source_images": [{"id": "test_image_1", "url": image_url}],
    }

    assert pipeline_request.model_dump() == expected_dict


def test_pipeline_moth_nonmoth(client, mock_image_request):
    image_path = get_test_image_path()
    image_url = mock_image_request(image_path)
    requests.get(image_url)
    pipeline_request = PipelineRequest(
        pipeline=PipelineChoice["moth_nonmoth"],
        source_images=[SourceImageRequest(id="test_image_1", url=image_url)],
    )
    response = client.post(
        "/pipeline/process/",
        json=pipeline_request.model_dump(),
    )
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) == 1
    result = response.json()["results"][0]
    assert "detections" in result
    assert "classifications" in result
