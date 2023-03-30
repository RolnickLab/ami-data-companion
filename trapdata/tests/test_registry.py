from trapdata.ml.models import binary_classifiers, object_detectors, species_classifiers


def run():
    for model in object_detectors + binary_classifiers + species_classifiers:
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


if __name__ == "__main__":
    run()
