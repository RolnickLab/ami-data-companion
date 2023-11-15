import gradio as gr
import PIL.Image
import PIL.ImageDraw
from rich import print

from .models.localization import MothDetector
from .schemas import SourceImage


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
    for source_image, result_source_image in zip(source_images, detector.results):
        source_image.detections = result_source_image.detections

    # ASSUME SINGLE IMAGE
    assert len(source_images) == 1
    source_image = source_images[0]
    source_image.open(raise_exception=True)
    assert source_image.pil is not None

    # Create PIL Images:
    print("Rendering crops")
    crops = []
    for bbox in source_image.detections:
        coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        crop = source_image.pil.crop(coords)
        crops.append((crop, "Unknown"))

    # Draw bounding boxes on image
    print("Drawing bounding boxes on source image")
    canvas = PIL.ImageDraw.Draw(source_image.pil)
    for bbox in source_image.detections:
        # Draw rectangle on PIL Image
        coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        canvas.rectangle(coords, outline="green", width=8)

    print("Returning results")
    return source_image.pil, crops

    # classifier = MothClassifier(detections=detections)
    # classifier.run()


app = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="filepath")],
    outputs=["image", "gallery"],
)  # json output type does not work

if __name__ == "__main__":
    app.launch()
