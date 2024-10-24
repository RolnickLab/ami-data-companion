import typing

import gradio as gr
import PIL.Image
import PIL.ImageDraw
from rich import print

from .models.classification import (
    MothClassifier,
    MothClassifierBinary,
    MothClassifierGlobal,
    MothClassifierPanama,
    MothClassifierPanama2024,
    MothClassifierQuebecVermont,
    MothClassifierTuringAnguilla,
    MothClassifierTuringCostaRica,
    MothClassifierUKDenmark,
)
from .models.localization import MothDetector
from .schemas import SourceImage
from .tests import get_test_images


def predict(*img_paths, Classifier: typing.Type[MothClassifier]):
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

    if len(source_images) == 1:
        detector = MothDetector(
            source_images=source_images,
            single=True,
        )
    else:
        detector = MothDetector(
            source_images=source_images,
            single=False,
        )
    detector.run()
    all_detections = detector.results

    # Filter out non-moths
    filter = MothClassifierBinary(
        source_images=source_images,
        detections=all_detections,
        filter_results=True,
        batch_size=20,
    )
    filter.run()
    filtered_detections = filter.results

    classifier = Classifier(
        source_images=source_images,
        detections=filtered_detections,
        batch_size=20,
    )
    classifier.run()

    # ASSUME SINGLE IMAGE
    assert len(source_images) == 1
    source_image = source_images[0]
    source_image.open(raise_exception=True)
    assert source_image._pil is not None

    # Draw bounding boxes on image
    print("Drawing bounding boxes on source image")
    annotated_image = source_image._pil.copy()
    canvas = PIL.ImageDraw.Draw(annotated_image)
    colors = {
        MothClassifierBinary.positive_binary_label: "green",
        MothClassifierBinary.negative_binary_label: "red",
    }
    default_color = "black"
    for pred in filtered_detections:
        # Draw rectangle on PIL Image
        assert pred.bbox is not None
        coords = (
            pred.bbox.x1,
            pred.bbox.y1,
            pred.bbox.x2,
            pred.bbox.y2,
        )
        # First classification is the binary one
        binary_label = pred.classifications[0].labels[0]
        color = colors.get(binary_label, default_color)
        canvas.rectangle(coords, outline=color, width=8)

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
        labels = pred.classifications[-1].labels
        scores = pred.classifications[-1].scores
        top_label_with_score = f"{labels[0]} ({scores[0]:.2f})"
        crops.append((crop, top_label_with_score))

    print("Returning results")
    return annotated_image, crops


from functools import partial


def make_interface(Classifier: MothClassifier, example_images_subdirs: list[str] = []):
    return gr.Interface(
        title=Classifier.name,
        description=Classifier.description,
        fn=partial(predict, Classifier=Classifier),  # type: ignore
        inputs=[gr.Image(type="filepath", label="Source Image")],
        outputs=[
            gr.Image(label="Annotated Source Image"),
            gr.Gallery(label="Classified Crops"),
        ],
        examples=[
            img.filepath for img in get_test_images(subdirs=example_images_subdirs)
        ],
    )


app = gr.TabbedInterface(
    [
        make_interface(MothClassifierPanama, ["panama"]),
        make_interface(MothClassifierPanama2024, ["panama"]),
        make_interface(MothClassifierUKDenmark, ["denmark"]),
        make_interface(MothClassifierQuebecVermont, ["vermont"]),
        make_interface(MothClassifierTuringCostaRica, ["costarica"]),
        make_interface(MothClassifierTuringAnguilla, ["anguilla"]),
        make_interface(
            MothClassifierGlobal,
            ["vermont", "panama", "denmark", "costarica", "anguilla"],
        ),
    ],
    [
        "Panama 2023",
        "Panama 2024",
        "UK/Denmark",
        "Quebec/Vermont",
        "Costa Rica",
        "Anguilla",
        "Global",
    ],
)

if __name__ == "__main__":
    app.launch(
        debug=True,
        show_error=True,
        show_api=False,
        server_name="0.0.0.0",
        server_port=7861,
    )
