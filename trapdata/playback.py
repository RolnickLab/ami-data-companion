import json
import pathlib
import random
import sys


from PIL import Image as PImage

import kivy

kivy.require("2.1.0")


from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import Rectangle, Color, Canvas, Line, Ellipse
from kivy.core.window import Window
from kivy.properties import StringProperty, ListProperty, ObjectProperty
from kivy.uix.screenmanager import Screen


from .utils import *


Builder.load_file(str(pathlib.Path(__file__).parent / "playback.kv"))


def choose_sample(images, direction, last_sample=None):

    if last_sample:
        last_sample = pathlib.Path(last_sample).name
        last_idx = images.index(last_sample)
    else:
        last_idx = 0

    if direction > 0:
        idx = last_idx + 1
    elif direction < 0:
        idx = last_idx - 1
    print("Sample index:", idx)
    sample = images[idx % len(images)]
    print("Last sample :", last_sample)
    print("Loading sample :", sample)
    return sample


def get_sequential_sample(direction, source_dir, last_sample=None):
    source_dir = pathlib.Path(source_dir)
    annotation_files = find_annotations(source_dir)
    if annotation_files:
        annotations = json.load(open(annotation_files[0]))
    else:
        raise Exception(f"No annotations found in directory: {source_dir}")

    if "images" in annotations:
        # This is from MegaDetector

        samples = list(annotations["images"])
        # sample = random.choice(list(annotations["images"]))
        samples = {s["file"]: s for s in samples}
        filenames = list(samples.keys())
        img_name = choose_sample(filenames, direction, last_sample)
        img_path = source_dir / img_name
        sample = samples[img_name]
        bboxes = []
        for detection in sample["detections"]:
            img_width, img_height = PImage.open(img_path).size
            print("PIL image:", img_width, img_height)
            x, y, width, height = detection["bbox"]
            x1 = x * img_width
            y1 = y * img_height
            x2 = (width * img_width) + x1
            y2 = (height * img_height) + y1
            bbox = [x1, y1, x2, y2]
            print("MegaDetector bbox:", detection["bbox"])
            print("MegaDetector bbox converted:", bbox)
            bboxes.append(bbox)
    else:
        # Assume this is Aditya's format

        images = list(annotations.keys())
        # sample = random.choice(list(annotations.keys()))
        sample = choose_sample(images, direction, last_sample)
        img_path = source_dir / sample
        bboxes = annotations[sample][0]
        labels = annotations[sample][1]

    return img_path, bboxes


class AnnotatedImage(Widget):
    # image_path = StringProperty()
    # bboxes = ListProperty()

    def __init__(self, **kwargs):

        self.image_path = kwargs.pop("image_path")
        self.bboxes = kwargs.pop("bboxes")

        super(AnnotatedImage, self).__init__(**kwargs)

        # Arranging Canvas
        with self.canvas:

            # Update the canvas as the screen size change
            self.bind(pos=self.update_rect, size=self.update_rect)

        self.draw()

    # update function which makes the canvas adjustable.
    def update_rect(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size
        self.draw()

    def draw(self, *args):
        self.canvas.clear()
        with self.canvas:
            img = Image(
                source=str(self.image_path.absolute()),
                pos=(0, 0),
                size=self.size,
                pos_hint={"bottom": 0},
                keep_ratio=False,
                allow_stretch=True,
            )
            print("Image sizes:")
            print(img.width, img.size, img.norm_image_size, img.texture_size)
            print()
            win_width, win_height = img.norm_image_size
            img_width, img_height = img.texture_size
            x_scale = win_width / img_width
            y_scale = win_height / img_height
            print("x scale:", x_scale, "y_scale:", y_scale)
            self.bg = img
            self.bbox_widgets = []
            for i, box in enumerate(self.bboxes):
                # Red box around canvas for debugging
                # Color(1, 0, 0, 1)
                # self.bbox_widgets.append(
                #     Line(rectangle=(0, 0, win_width, win_height), width=2)
                # )
                # print("bbox#", i)

                color = [random.random() for _ in range(3)]
                color.append(0.8)  # alpha
                Color(*color)
                x1, y1, x2, y2 = box

                w = x2 - x1
                h = y2 - y1
                # print("original dims:", w, h)

                # Reference from bottom left instead of top left
                y1 = img_height - y1
                y2 = img_height - y2

                # Scale bbox to match on-screen scale of image
                x1 *= x_scale
                y1 *= y_scale
                x2 *= x_scale
                y2 *= y_scale

                w2 = w * x_scale
                h2 = h * y_scale
                # print("new dims:", w2, h2)

                w2 = x2 - x1
                h2 = y1 - y2
                # print("new dims by coord:", w2, h2)

                rect = (x1, y2, x2, y1)
                # print(box, rect)
                self.bbox_widgets.append(
                    Line(points=[x1, y1, x1, y2, x2, y2, x2, y1, x1, y1], width=2)
                )
                self.bbox_widgets.append(
                    Label(text=str(i), center=(x1, y2 - 20), color=color)
                )


class ImagePlaybackLayout(BoxLayout):
    pass


class ImagePlaybackScreen(Screen):
    source_dir = ObjectProperty()

    def on_source_dir(self, instance, value):
        self.current_sample = None
        preview = self.ids.playback_layout.ids.image_preview
        preview.reset()
        preview.next_sample()


class BBox(BoxLayout):
    pass


class PreviewWindow(RelativeLayout):
    # @TODO save current sample for each directory, so we keep our place
    current_sample = ObjectProperty(allownone=True)

    def reset(self):
        self.current_sample = None
        self.clear_widgets()

    def load_sample(self, img_path, annotations=None):
        bboxes = annotations
        self.clear_widgets()
        cvs = AnnotatedImage(
            image_path=img_path, bboxes=bboxes, size=self.size, pos_hint={"bottom": 0}
        )
        self.current_sample = img_path
        self.add_widget(cvs)

    def next_sample(self):
        img_path, bboxes = get_sequential_sample(
            direction=1,
            source_dir=self.parent.parent.source_dir,
            last_sample=self.current_sample,
        )
        print("Next!", img_path)
        self.load_sample(img_path, annotations=bboxes)

    def prev_sample(self):
        img_path, bboxes = get_sequential_sample(
            direction=-1,
            source_dir=self.parent.parent.source_dir,
            last_sample=self.current_sample,
        )
        print("Prev!", img_path)
        self.load_sample(img_path, annotations=bboxes)


class Controls(BoxLayout):
    pass


class ImageOverlayApp(App):
    def build(self):
        self.title = "Image bbox overlay test"
        # This just loads an example dir for testing
        img_dir = choose_root_directory(cache=False)
        layout = ImagePlaybackScreen(source_dir=img_dir, name="playback")
        Window.clearcolor = (0, 1, 0, 1)

        label = Label(size_hint=(1, 0.05))
        Window.bind(mouse_pos=lambda w, p: setattr(label, "text", str(p)))
        layout.add_widget(label)

        return layout


if __name__ == "__main__":
    ImageOverlayApp().run()
