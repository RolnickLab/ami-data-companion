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
from kivy.clock import Clock
from kivy.properties import (
    StringProperty,
    ListProperty,
    ObjectProperty,
    NumericProperty,
)
from kivy.uix.screenmanager import Screen


from .utils import *


Builder.load_file(str(pathlib.Path(__file__).parent / "playback.kv"))


class AnnotatedImage(Widget):
    image_path = ObjectProperty()
    annotations = ListProperty()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

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

        if not self.image_path.exists():
            logger.error(f"Image not found: {self.image_path}")
            return None

        with self.canvas:
            img = Image(
                source=str(self.image_path),
                pos=(0, 0),
                size=self.size,
                pos_hint={"bottom": 0},
                keep_ratio=False,
                allow_stretch=True,
            )
            win_width, win_height = img.norm_image_size
            img_width, img_height = img.texture_size
            x_scale = win_width / img_width
            y_scale = win_height / img_height
            self.bg = img
            self.bbox_widgets = []
            for i, annotation in enumerate(self.annotations):
                # Red box around canvas for debugging
                # Color(1, 0, 0, 1)
                # self.bbox_widgets.append(
                #     Line(rectangle=(0, 0, win_width, win_height), width=2)
                # )
                # print("bbox#", i)

                if annotation.binary_label == POSITIVE_BINARY_LABEL:
                    color = POSITIVE_COLOR
                else:
                    color = NEGATIVE_COLOR

                # color = [random.random() for _ in range(3)]
                # color.append(0.8)  # alpha
                Color(*color)

                if not annotation.bbox:
                    logging.warn(
                        f"No bbox for detected object {annotation.id}. Skipping."
                    )
                    continue

                x1, y1, x2, y2 = annotation.bbox

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

                if annotation.binary_label == NEGATIVE_BINARY_LABEL:
                    label_text = ""

                elif annotation.specific_label:
                    # If there is a binary label and it's nonmoth, don't show
                    # the specific label, even if one exists.
                    if annotation.specific_label_score:
                        score = round(annotation.specific_label_score * 100, 1)
                        label_text = f"{annotation.specific_label} ({score}%)"
                    else:
                        label_text = f"{annotation.specific_label}"

                elif annotation.binary_label:
                    if annotation.binary_label_score:
                        score = round(annotation.binary_label_score * 100, 1)
                        label_text = f"{annotation.binary_label} ({score}%)"
                    else:
                        label_text = f"{annotation.binary_label}"

                else:
                    label_text = ""

                self.bbox_widgets.append(
                    Label(
                        text=label_text,
                        center=((x1 + w2 / 2), y2 - 20),
                        color=color,
                        bold=True,
                        halign="center",
                    )
                )


DEFAULT_FPS = 2


class ImagePlaybackScreen(Screen):
    monitoring_session = ObjectProperty()
    image_ids = ListProperty()
    fps = NumericProperty(defaultvalue=DEFAULT_FPS)
    clock = ObjectProperty(allownone=True)

    def reload(self, ms):
        self.current_sample = None
        self.image_ids = [img.id for img in get_monitoring_session_image_ids(ms)]
        preview = self.ids.image_preview
        preview.reset()
        preview.next_sample()
        self.pause()

    def on_monitoring_session(self, instance, value):
        ms = value
        self.reload(ms)

    def _play_callback(self, dt):
        # @TODO stop at last frame
        self.ids.image_preview.next_sample()

    def play(self):
        logger.debug("Starting playback")
        if self.clock:
            # Already running, increase framerate
            Clock.unschedule(self.clock)
            self.fps += 2
        self.clock = Clock.schedule_interval(self._play_callback, 1 / self.fps)
        self.ids.play_button.text = f"Play ({self.fps} FPS)"

    def pause(self):
        logger.debug("Stopping playback")
        if self.clock:
            Clock.unschedule(self.clock)
            self.clock = None
            self.fps = DEFAULT_FPS
        self.ids.play_button.text = f"Play"

    def exit(self):
        self.manager.current = "menu"

    def on_leave(self, *args):
        self.pause()


class BBox(BoxLayout):
    pass


class PreviewWindow(RelativeLayout):
    # @TODO save current sample for each directory, so we keep our place
    current_sample = ObjectProperty(allownone=True)

    def reset(self):
        self.current_sample = None
        self.clear_widgets()

    def load_sample(self, image_id):
        ms = self.parent.parent.monitoring_session
        # Refetch image with associated detected objects
        image = get_image_with_objects(ms, image_id)
        self.clear_widgets()
        cvs = AnnotatedImage(
            image_path=image.absolute_path(ms.base_directory),
            annotations=image.detected_objects,
            size=self.size,
            pos_hint={"bottom": 0},
        )
        self.current_sample = image
        self.add_widget(cvs)

    def next_sample(self, *args):
        image_id = get_sequential_sample(
            direction=1,
            images=self.parent.parent.image_ids,
            last_sample=self.current_sample.id if self.current_sample else None,
        )
        self.load_sample(image_id)

    def prev_sample(self, *args):
        image_id = get_sequential_sample(
            direction=-1,
            images=self.parent.parent.image_ids,
            last_sample=self.current_sample.id if self.current_sample else None,
        )
        self.load_sample(image_id)


class ImageOverlayApp(App):
    def build(self):
        self.title = "Image bbox overlay test"
        # This just loads an example dir for testing
        img_dir = choose_directory(cache=False)
        layout = ImagePlaybackScreen(source_dir=img_dir, name="playback")
        Window.clearcolor = (0, 1, 0, 1)

        label = Label(size_hint=(1, 0.05))
        Window.bind(mouse_pos=lambda w, p: setattr(label, "text", str(p)))
        layout.add_widget(label)

        return layout


if __name__ == "__main__":
    ImageOverlayApp().run()
