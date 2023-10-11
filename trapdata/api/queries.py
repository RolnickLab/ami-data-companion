import logging

from . import settings
from .auth import get_session
from .schemas import IncomingSourceImage

logger = logging.getLogger(__name__)


def save_detected_objects(
    source_image_ids: list[int], detected_objects_data: list[dict], *args, **kwargs
):
    logger.info(f"Saving {len(source_image_ids)} detected objects via API")
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
            data["detection_algorithm_id"] = 1  # detected_object["model_name"]
            resp = get_session().post(settings.api_base_url + path, json=data)
            resp.raise_for_status()
            data = resp.json()
            logger.info(
                f"Saved detected object {data['details']} with width {data['width']} and height {data['height']}"
            )
            responses[source_image_id] = data
    return responses


def get_next_source_images(num: int, *args, **kwargs) -> list[IncomingSourceImage]:
    path = "captures/"
    args = {
        "limit": num,
        "deployment": 9,
        "event": 34,
        "has_detections": False,
    }  # last_processed__isnull=True
    url = settings.api_base_url + path
    resp = get_session().get(url, params=args)
    resp.raise_for_status()
    data = resp.json()
    source_images = [IncomingSourceImage(**item) for item in data["results"]]
    return source_images


def get_source_image_count(*args, **kwargs) -> int:
    path = "captures/"
    args = {
        "deployment": 9,
        "event": 34,
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
