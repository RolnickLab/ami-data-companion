import logging

import structlog

# structlog.configure(
#     wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
# )

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
)

logger = structlog.get_logger()
logging.disable(logging.CRITICAL)

# import logging
# from rich.logging import RichHandler
#
# FORMAT = "%(message)s"
# logging.basicConfig(
#     level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
# )
#
# logger= logging.getLogger("rich")
