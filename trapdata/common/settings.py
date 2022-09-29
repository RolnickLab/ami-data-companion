import pathlib

from .logs import logger


def cache_dir(path=None):
    # If fails, use temp dir?
    # d = tempfile.TemporaryDirectory(delete=False)
    path = path or ".cache"
    d = pathlib.Path(".cache")
    d.mkdir(exist_ok=True)
    return d


def save_setting(key, val):
    """
    >>> save_setting("last_test", "now")
    'now'
    >>> read_setting("last_test")
    'now'
    """
    f = cache_dir() / key
    logger.debug(f"Writing to cache: {f}")
    f.write_text(val)
    return val


def read_setting(key):
    f = cache_dir() / key
    logger.debug(f"Checking cache: {f}")
    if f.exists():
        return f.read_text()
    else:
        return None


def delete_setting(key):
    f = cache_dir() / key
    logger.debug(f"Deleting cache: {f}")
    if f.exists():
        return f.unlink()
    else:
        return None
