SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg")

POSITIVE_BINARY_LABEL = "moth"
NEGATIVE_BINARY_LABEL = "nonmoth"
NULL_DETECTION_LABELS = [NEGATIVE_BINARY_LABEL]
TRACKING_COST_THRESHOLD = 1.0

POSITIVE_COLOR = [0, 100 / 255, 1, 1]  # Blue
# POSITIVE_COLOR = [1, 0, 162 / 255, 1]  # Pink
# NEUTRAL_COLOR = [1, 1, 1, 0.5]  # White
# NEUTRAL_COLOR = [1, 0, 162 / 255, 0.2]  # Pink, semi-transparent
NEUTRAL_COLOR = [0, 100 / 255, 1, 0.4]  # Blue
NEGATIVE_COLOR = [1, 1, 1, 0]  # Transparent

SUMMARY_REFRESH_SECONDS = 5

IMAGE_BASE_URL = "https://object-arbutus.cloud.computecanada.ca/ami-trapdata/"
