import logging
import sys

logger = logging.getLogger().getChild(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# Use different config if running tests
if hasattr(sys.modules["__main__"], "_SpoofOut"):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
