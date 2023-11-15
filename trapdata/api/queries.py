from trapdata import logger

from . import settings
from .auth import get_session
from .schemas import SourceImage

TEMPORARY_DEPLOYMENT_ID = 9
TEMPORARY_EVENT_ID = 34
TEMPORARY_COLLECTION_ID = 4


def fetch_source_image_data(id: int):
    path = f"captures/{id}/"
    url = settings.api_base_url + path
    resp = get_session().get(url)
    resp.raise_for_status()
    data = resp.json()
    return SourceImage(**data)


# @TODO If this is being initiated from a POST request with the source image info, then we can
# return the results in the same response. At least single images.
def save_detected_objects(
    source_image_ids: list[int], detected_objects_data: list[dict], *args, **kwargs
):
    logger.info(f"Saving {len(source_image_ids)} detected objects via API")
    print(f"Saving {len(source_image_ids)} detected objects via API")
    responses = {}
    path = "detections/"

    for source_image_id, detected_objects in zip(
        source_image_ids, detected_objects_data
    ):
        for detected_object in detected_objects:
            data = {}
            data["bbox"] = detected_object["bbox"]
            # data["source_image_id"] = source_image_id
            data["source_image"] = (
                settings.api_base_url + f"captures/{source_image_id}/"
            )
            data[
                "detection_algorithm_id"
            ] = 2  # https://api.dev.insectai.org/api/v2/ml/algorithms/2/
            resp = get_session().post(settings.api_base_url + path, json=data)
            resp.raise_for_status()
            data = resp.json()
            logger.info(
                f"Saved detected object {data['details']} with width {data['width']} and height {data['height']}"
            )
            responses[source_image_id] = data
    return responses


def get_next_source_images(num: int, *args, **kwargs) -> list[SourceImage]:
    path = "captures/"
    args = {
        "limit": num,
        # "deployment": TEMPORARY_DEPLOYMENT_ID,
        # "event": TEMPORARY_EVENT_ID,
        "collections": TEMPORARY_COLLECTION_ID,
        "has_detections": False,
        "order": "?",
    }  # last_processed__isnull=True
    url = settings.api_base_url + path
    resp = get_session().get(url, params=args)
    resp.raise_for_status()
    data = resp.json()
    source_images = [SourceImage(**item) for item in data["results"]]
    return source_images


def get_source_image_count(*args, **kwargs) -> int:
    path = "captures/"
    args = {
        # "deployment": TEMPORARY_DEPLOYMENT_ID,
        # "event": TEMPORARY_EVENT_ID,
        "collections": TEMPORARY_COLLECTION_ID,
        "limit": 1,
        "has_detections": False,
    }
    url = settings.api_base_url + path
    resp = get_session().get(url, params=args)
    print(resp.url)
    resp.raise_for_status()
    data = resp.json()
    count = data["count"]
    logger.info(f"Images remaining to process: {count}")
    return count


def get_totals(project_id: int, *args, **kwargs):
    path = "status/summary"
    params = {"project": project_id}
    url = settings.api_base_url + path
    resp = get_session().get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data


class QueueManager:
    pass
