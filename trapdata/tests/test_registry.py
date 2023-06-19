from trapdata.ml.models import binary_classifiers, object_detectors, species_classifiers


def test_model_registry():
    # @TODO need an assertion to test
    for model in (
        list(object_detectors.values())
        + list(binary_classifiers.values())
        + list(species_classifiers.values())
    ):
        print(
            "\n".join(
                str(item)
                for item in [
                    model,
                    model.name,
                    model.type,
                    model.description,
                    "\n",
                ]
            )
        )
    assert True


if __name__ == "__main__":
    test_model_registry()
