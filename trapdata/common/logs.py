import logging

import structlog


structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
)

logger = structlog.get_logger()
