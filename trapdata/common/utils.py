def get_sequential_sample(direction, images, last_sample=None):
    if not images:
        return None

    if last_sample:
        last_idx = images.index(last_sample)
    else:
        last_idx = 0

    if direction > 0:
        idx = last_idx + 1
    elif direction < 0:
        idx = last_idx - 1

    sample = images[idx % len(images)]
    return sample


def slugify(s):
    return s.replace(" ", "_").lower()


def bbox_area(bbox):
    """
    Return the area of a bounding box.

    Bounding boxes are assumed to be in the format:
    [(top-left-coordinate-pair), (bottom-right-coordinate-pair)]
    or: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    area = (y2 - y1) * (x2 - x1)
    return area
