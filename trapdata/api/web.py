import json

import gradio as gr

from .models.localization import MothDetector
from .schemas import SourceImage


def predict(*img_paths):
    # img_paths = img_paths * 4  # Test batch size
    source_images = [
        SourceImage(id=i, filepath=img_path) for i, img_path in enumerate(img_paths)
    ]

    if len(source_images) == 1:
        detector = MothDetector(source_images=source_images, single=True)
    else:
        detector = MothDetector(source_images=source_images, batch_size=2, single=False)

    detector.run()
    results = json.dumps(detector.results, indent=2)
    return results


app = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="filepath")],
    outputs="text",
)  # json output type does not work

if __name__ == "__main__":
    app.launch()
