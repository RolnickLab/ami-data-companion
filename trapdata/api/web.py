import json

import gradio as gr

from .models.localization import MothDetector
from .schemas import IncomingSourceImage


def predict(*img_paths):
    source_images = [
        IncomingSourceImage(id=i, filepath=img_path)
        for i, img_path in enumerate(img_paths)
    ]

    detector = MothDetector(source_images=source_images, single=True)
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
