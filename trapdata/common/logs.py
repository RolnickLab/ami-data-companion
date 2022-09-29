import logging
import sys

logger = logging.getLogger(__name__)
# logger = logging.getLogger().getChild(__name__)
formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


# Use different config if running tests
if hasattr(sys.modules["__main__"], "_SpoofOut"):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
