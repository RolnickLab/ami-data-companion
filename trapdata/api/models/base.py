class APIInferenceBaseClass:
    """
    Override methods and properties in the InferenceBaseClass that
    are needed or not needed for the API version.
    """

    def __init__(self, *args, **kwargs):
        # Don't need to set these for API version
        kwargs["db_path"] = None
        kwargs["image_base_path"] = None
        super().__init__(*args, **kwargs)
