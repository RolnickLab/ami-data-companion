import sentry_sdk

sentry_sdk.init(
    dsn="https://d2f65f945fe343669bbd3be5116d5922@o4503927026876416.ingest.sentry.io/4503927029497856",
    traces_sample_rate=1.0,
)

import multiprocessing

from .common.logs import logger
from .common import utils
from .common import constants
from .db.models.images import TrapImage
from .db.models.detections import DetectedObject
from .db.models.events import MonitoringSession


__all__ = [
    logger,
    utils,
    constants,
    TrapImage,
    DetectedObject,
    MonitoringSession,
]

# Required for PyTorch. Default on Windows.
multiprocessing.set_start_method("fork")
