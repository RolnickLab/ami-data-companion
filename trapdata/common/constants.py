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

# Default location of the public object store that holds the model weights, label maps
# and sample trap images. Deployments can download from elsewhere by setting
# model_base_url and image_base_url (AMI_MODEL_BASE_URL, AMI_IMAGE_BASE_URL), which
# default to buckets under this URL. The Swift path form is used because the equivalent
# S3 path form puts a "<tenant>:" prefix on the bucket name, and the colon trips some
# URL parsers and caches.
OBJECT_STORE_BASE_URL = (
    "https://object-arbutus.alliancecan.ca/swift/v1/"
    "AUTH_3c987b8fc90743469d42899b1fdb48eb/"
)
