import pathlib
import datetime

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import Color, Line
from kivy.clock import Clock
from kivy.properties import (
    ListProperty,
    ObjectProperty,
    NumericProperty,
)
from kivy.uix.screenmanager import Screen

from trapdata import logger
from trapdata import constants
from trapdata.db.models.events import get_monitoring_session_image_ids
from trapdata.db.models.images import get_image_with_objects, completely_classified
from trapdata.db.models.detections import (
    get_detections_for_image,
    get_species_for_image,
)
from trapdata.db.models.queue import add_image_to_queue
from trapdata.common.utils import get_sequential_sample


kivy.require("2.1.0")


Builder.load_file(str(pathlib.Path(__file__).parent / "playback.kv"))


class AnnotatedImage(Widget):
    image_path = ObjectProperty()
    annotations = ListProperty()
    image = ObjectProperty()

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

                if annotation.binary_label == constants.POSITIVE_BINARY_LABEL:
                    color = constants.POSITIVE_COLOR
                # elif annotation.binary_label == NEGATIVE_BINARY_LABEL:
                #     color = NEGATIVE_COLOR
                # else:
                #     color = NEUTRAL_COLOR
                else:
                    color = constants.NEGATIVE_COLOR

                # color = [random.random() for _ in range(3)]
                # color.append(0.8)  # alpha
                Color(*color)

                if not annotation.bbox:
                    logger.warn(
                        f"No bbox for detected object {annotation.id}. Skipping."
                    )
                    continue

                x1, y1, x2, y2 = annotation.bbox

                w = x2 - x1
                # h = y2 - y1
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
                # h2 = h * y_scale
                # print("new dims:", w2, h2)

                w2 = x2 - x1
                # h2 = y1 - y2
                # print("new dims by coord:", w2, h2)

                # rect = (x1, y2, x2, y1)

                self.bbox_widgets.append(
                    Line(points=[x1, y1, x1, y2, x2, y2, x2, y1, x1, y1], width=2)
                )

                if annotation.binary_label == constants.NEGATIVE_BINARY_LABEL:
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

            app = App.get_running_app()
            detections = get_detections_for_image(app.db_path, self.image.id).count()
            species = get_species_for_image(app.db_path, self.image.id).count()
            if self.image.last_processed:
                time_delta = f"{(datetime.datetime.now() - self.image.last_processed).seconds}s ago"
            else:
                time_delta = "Never"

            label_text = (
                f"{self.image.timestamp} | "
                f"In Queue: {self.image.in_queue} | "
                f"Detections: {detections} | "
                f"Species: {species} | "
                f"Complete: {completely_classified(app.db_path, self.image.id)} | "
                f"Processed: {time_delta}"
            )
            label = Label(
                text=label_text,
                color=[0, 0, 0, 1],
                halign="left",
                size=self.size,
                center=(self.size[0] / 2, 20),
                font_size="14sp",
            )
            # label.center = (label.center[0] + label.width * 2, 30)
            self.add_widget(label)


DEFAULT_FPS = 2


class ImagePlaybackScreen(Screen):
    monitoring_session = ObjectProperty()
    image_ids = ListProperty()
    fps = NumericProperty(defaultvalue=DEFAULT_FPS)
    clock = ObjectProperty(allownone=True)

    def reload(self, ms):
        self.current_sample = None
        app = App.get_running_app()
        self.image_ids = [
            img.id for img in get_monitoring_session_image_ids(app.db_path, ms)
        ]
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
        self.ids.play_button.text = "Play"

    def exit(self):
        self.manager.current = "menu"

    def on_leave(self, *args):
        self.pause()


class BBox(BoxLayout):
    pass


class PreviewWindow(RelativeLayout):
    # @TODO save current sample for each directory, so we keep our place
    current_sample = ObjectProperty(allownone=True)
    queue_clock = ObjectProperty()

    def reset(self):
        self.current_sample = None
        self.clear_widgets()

    def refresh(self, *args):
        self.load_sample(self.current_sample.id)

    def load_sample(self, image_id):
        ms = self.parent.parent.monitoring_session
        # Refetch image with associated detected objects
        app = App.get_running_app()
        image = get_image_with_objects(app.db_path, ms, image_id)
        self.clear_widgets()
        cvs = AnnotatedImage(
            image_path=image.absolute_path,
            annotations=image.detected_objects,
            size=self.size,
            pos_hint={"bottom": 0},
            image=image,
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

    def add_sample_to_queue(self):
        app = App.get_running_app()
        add_image_to_queue(app.db_path, self.current_sample.id)
        app.start_queue()
        # @TODO kill clock when all objects have been classified!
        Clock.schedule_interval(self.refresh, 1)

    def button_text(self):
        if self.current_sample and self.current_sample.in_queue:
            return "(In Queue)"
        else:
            return "Add to Queue"
