import gradio as gr
import PIL.Image
import PIL.ImageDraw
from rich import print

from .models.classification import MothClassifier
from .models.localization import MothDetector
from .schemas import Detection, SourceImage


def predict(*img_paths):
    # img_paths = img_paths * 4  # Test batch size
    source_images = [
        SourceImage(id=str(i), filepath=img_path)
        for i, img_path in enumerate(img_paths)
    ]
    # Print size of each source image
    for source_image in source_images:
        source_image.open(raise_exception=True)
        print(source_image, source_image.pil.size)

    if len(source_images) == 1:
        detector = MothDetector(source_images=source_images, single=True)
    else:
        detector = MothDetector(source_images=source_images, batch_size=2, single=False)

    detector.run()
    print(detector.results)
    assert len(detector.results) == len(source_images)

    # Update source images with detections
    # This is only necessary if rendering before Detection objects are created
    for source_image, result_source_image in zip(source_images, detector.results):
        source_image.detections = result_source_image.detections

    # Create detection objects
    detections = []
    for test_image, result_image in zip(source_images, detector.results):
        for bbox in result_image.detections:
            detection = Detection(source_image=test_image, bbox=bbox)
            detections.append(detection)

    classifier = MothClassifier(detections=detections)
    classifier.run()

    # ASSUME SINGLE IMAGE
    assert len(source_images) == 1
    source_image = source_images[0]
    source_image.open(raise_exception=True)
    assert source_image.pil is not None

    # Create PIL Images:
    # print("Rendering crops")
    # crops = []
    # for bbox in source_image.detections:
    #     coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    #     crop = source_image.pil.crop(coords)
    #     crops.append((crop, "Unknown"))

    # Create PIL Images:
    print("Rendering crops")
    crops = []
    for detection in classifier.results:
        bbox = detection.bbox
        assert bbox is not None
        coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        crop = source_image.pil.crop(coords)
        top_label_with_score = f"{detection.labels[0]} ({detection.scores[0]:.2f})"
        crops.append((crop, top_label_with_score))

    # Draw bounding boxes on image
    print("Drawing bounding boxes on source image")
    annotated_image = source_image.pil.copy()
    canvas = PIL.ImageDraw.Draw(annotated_image)
    for bbox in source_image.detections:
        # Draw rectangle on PIL Image
        coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        canvas.rectangle(coords, outline="green", width=8)

    print("Returning results")
    return annotated_image, crops


app = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="filepath")],
    outputs=["image", "gallery"],
)  # json output type does not work

if __name__ == "__main__":
    app.launch()
