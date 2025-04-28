import logging
import os

import structlog


def get_logger():
    """
    Get a logger instance with the specified log level.

    Set a log level using the AMI_LOG_LEVEL environment variable. For example:

    ```
    export AMI_LOG_LEVEL=DEBUG
    ami api
    ```

    or

    ```
    AMI_LOG_LEVEL=CRITICAL ami api
    ```

    @TODO
    It would be ideal if we could configure a log level in the Settings class,
    but there are issues with circular imports.
    """
    log_level = logging.getLevelName(os.environ.get("AMI_LOG_LEVEL", "INFO"))

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    logger = structlog.get_logger()
    return logger


logger = get_logger()
