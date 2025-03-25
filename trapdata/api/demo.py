import datetime
import typing
from dataclasses import dataclass
from functools import partial

import gradio as gr
import PIL.Image
import PIL.ImageDraw
from rich import print

from .models.classification import (
    APIMothClassifier,
    MothClassifierGlobal,
    MothClassifierPanama,
    MothClassifierPanama2024,
    MothClassifierQuebecVermont,
    MothClassifierTuringCostaRica,
    MothClassifierUKDenmark,
    MultcoButterflySpeciesClassifier,
)
from .schemas import AlgorithmReference, BoundingBox, DetectionResponse, SourceImage
from .tests.test_models import get_test_images


@dataclass
class ClassifierChoice:
    key: str
    tab_title: str
    example_images_dir_names: list[str]
    classifier: type[APIMothClassifier]


CLASSIFIER_CHOICES = [
    ClassifierChoice(
        key=MothClassifierPanama.get_key(),
        tab_title="Panama 2023",
        example_images_dir_names=["panama"],
        classifier=MothClassifierPanama,
    ),
    ClassifierChoice(
        key=MothClassifierPanama2024.get_key(),
        tab_title="Panama 2024",
        example_images_dir_names=["panama"],
        classifier=MothClassifierPanama2024,
    ),
    ClassifierChoice(
        key=MothClassifierUKDenmark.get_key(),
        tab_title="UK/Denmark",
        example_images_dir_names=["denmark"],
        classifier=MothClassifierUKDenmark,
    ),
    ClassifierChoice(
        key=MothClassifierQuebecVermont.get_key(),
        tab_title="Quebec/Vermont",
        example_images_dir_names=["vermont"],
        classifier=MothClassifierQuebecVermont,
    ),
    ClassifierChoice(
        key=MothClassifierTuringCostaRica.get_key(),
        tab_title="Costa Rica",
        example_images_dir_names=["panama"],
        classifier=MothClassifierTuringCostaRica,
    ),
    ClassifierChoice(
        key=MultcoButterflySpeciesClassifier.get_key(),
        tab_title="Multco Butterfly",
        example_images_dir_names=["denmark"],
        classifier=MultcoButterflySpeciesClassifier,
    ),
    ClassifierChoice(
        key=MothClassifierGlobal.get_key(),
        tab_title="Global",
        example_images_dir_names=["vermont", "panama", "denmark"],
        classifier=MothClassifierGlobal,
    ),
]


def get_classifier_by_key(key: str) -> type[APIMothClassifier]:
    for choice in CLASSIFIER_CHOICES:
        if choice.key == key:
            return choice.classifier
    raise ValueError(f"Classifier with key '{key}' not found.")


def predict(*img_paths, Classifier: type[APIMothClassifier]):
    # img_paths = img_paths * 4  # Test batch size
    source_images = [
        SourceImage(id=str(i), filepath=img_path)
        for i, img_path in enumerate(img_paths)
    ]
    # Print size of each source image
    for source_image in source_images:
        source_image.open(raise_exception=True)
        assert source_image._pil is not None
        print(source_image, source_image._pil.size)

    # Create a single detection for each whole image
    whole_image_detections = []
    for _, source_image in enumerate(source_images):
        assert source_image._pil is not None
        width, height = source_image._pil.size
        # Create a bounding box for the entire image
        bbox = BoundingBox(x1=0, y1=0, x2=width, y2=height)
        # Create a detection for the whole image
        detection = DetectionResponse(
            source_image_id=source_image.id,
            bbox=bbox,
            algorithm=AlgorithmReference(
                name=Classifier.name, key=Classifier.get_key()
            ),
            timestamp=datetime.datetime.now(),
        )
        whole_image_detections.append(detection)

    # Pass the whole image detections directly to the species classifier
    classifier = Classifier(
        source_images=source_images,
        detections=whole_image_detections,
        batch_size=20,
    )
    classifier.run()

    all_labels = classifier.category_map
    assert all_labels, f"No labels found for {Classifier.name}"

    # ASSUME SINGLE IMAGE
    assert len(source_images) == 1
    source_image = source_images[0]
    source_image.open(raise_exception=True)
    assert source_image._pil is not None

    # Draw bounding boxes on image
    print("Drawing bounding boxes on source image")
    annotated_image = source_image._pil.copy()
    canvas = PIL.ImageDraw.Draw(annotated_image)
    default_color = "green"

    # Draw a single rectangle around the whole image
    for pred in whole_image_detections:
        # Draw rectangle on PIL Image
        assert pred.bbox is not None
        coords = (
            pred.bbox.x1,
            pred.bbox.y1,
            pred.bbox.x2,
            pred.bbox.y2,
        )
        canvas.rectangle(coords, outline=default_color, width=8)

    # Create PIL Images:
    print("Rendering crops")
    crops = []
    for pred in classifier.results:
        bbox = pred.bbox
        assert bbox is not None
        coords = (round(bbox.x1), round(bbox.y1), round(bbox.x2), round(bbox.y2))
        assert source_image._pil is not None
        crop = source_image._pil.crop(coords)
        # Last classification is the most specific one
        classification = pred.classifications[-1]
        best_score = max(classification.scores)
        best_label = classification.classification
        top_label_with_score = f"{best_label} ({best_score:.2f})"
        crops.append((crop, top_label_with_score))

    print("Returning results")
    return annotated_image, crops


def make_interface(
    Classifier: typing.Type[APIMothClassifier],
    example_images_subdirs: typing.Iterable[str] = (),
):
    return gr.Interface(
        title=Classifier.name,
        description=Classifier.description,
        fn=partial(predict, Classifier=Classifier),
        inputs=[gr.Image(type="filepath", label="Source Image")],
        outputs=[
            gr.Image(label="Annotated Source Image"),
            gr.Gallery(label="Classified Crops"),
        ],
        examples=[
            img.filepath for img in get_test_images(subdirs=example_images_subdirs)
        ],
    )


tab_interfaces = []
tab_titles = []
for choice in CLASSIFIER_CHOICES:
    tab_interfaces.append(
        make_interface(choice.classifier, choice.example_images_dir_names)
    )
    tab_titles.append(choice.tab_title)

app = gr.TabbedInterface(tab_interfaces, tab_titles)

if __name__ == "__main__":
    app.launch(
        debug=True,
        show_error=True,
        show_api=False,
        server_name="0.0.0.0",
        server_port=7861,
    )
