import pathlib
from functools import partial

import gradio as gr
import torch

from trapdata.ml.models.classification import (
    SimplePanamaClassifier,
    SimpleQuebecVermontClassifier,
    SimpleUKDenmarkClassifier,
)

EXAMPLES_DIR = pathlib.Path("trapdata/tests/images/cropped")

classifiers = [
    (
        "Panama",
        SimplePanamaClassifier(user_data_path=torch.hub.get_dir()),
        "panama",
    ),
    (
        "Quebec/Vermont",
        SimpleQuebecVermontClassifier(user_data_path=torch.hub.get_dir()),
        "vermont",
    ),
    (
        "UK/Denmark",
        SimpleUKDenmarkClassifier(user_data_path=torch.hub.get_dir()),
        "denmark",
    ),
]


def format_results(results):
    return {label: score for label, score in results}


def predict(image, classifier):
    results_batch = classifier.predict([image])
    return format_results(results_batch[0])


interfaces = [
    (
        gr.Interface(
            fn=partial(predict, classifier=classifier),
            inputs=gr.Image(
                type="pil",
                interactive=False,
                show_share_button=True,
                show_download_button=True,
                min_width=50,
            ),
            outputs=gr.Label(num_top_classes=3),
            examples=str(EXAMPLES_DIR / examples_subdir),
            title=classifier.name,
            description=classifier.description,
            analytics_enabled=True,
            allow_flagging="manual",
            live=False,
        ),
        name,
    )
    for name, classifier, examples_subdir in classifiers
]

interfaces, names = zip(*interfaces)

demo = gr.TabbedInterface(
    interface_list=interfaces,
    tab_names=names,
    theme=gr.themes.Default(),
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
