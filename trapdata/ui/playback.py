import pathlib
from typing import Optional

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
from trapdata.common.utils import format_timedelta_hours
from trapdata.db.base import get_session_class
from trapdata.db.models.events import (
    get_monitoring_session_image_ids,
    MonitoringSession,
)
from trapdata.db.models.images import TrapImage, get_image_with_objects
from trapdata.db.models.detections import (
    DetectedObject,
    get_object_counts_for_image,
    delete_objects_for_image,
    get_unique_objects_for_image,
)
from trapdata.db.models.queue import add_image_to_queue, clear_all_queues
from trapdata.common.utils import get_sequential_sample


kivy.require("2.1.0")


Builder.load_file(str(pathlib.Path(__file__).parent / "playback.kv"))


DEFAULT_FPS = 2


def split_label(text):
    return text.replace(" ", "\n")


def update_info_bar(widget: Label, image: TrapImage, stats: dict):
    if image.last_processed:
        last_processed = image.last_processed.strftime("%H:%M")
    else:
        last_processed = "Never"

    text = (
        "\n\n"
        f"In Queue: {image.in_queue} | "
        f"Objects: {stats.get('num_objects')} | "
        f"Detections: {stats.get('num_detections')} | "
        f"Species: {stats.get('num_species')} | "
        f"Complete: {stats.get('completely_classified')} | "
        f"Last Processed: {last_processed}\n"
        # '* "Process Now" button will clear the queue and interrupt any tracked sequences.'
    )

    widget.text = text


def update_title(widget: Label, image: TrapImage):
    text = f"{image.timestamp.strftime('%c')}"  # (image #{image.id})\n"

    widget.text = text
    # widget.bold = True


class AnnotatedImage(Widget):
    image_path: pathlib.Path = ObjectProperty()
    annotations: list[DetectedObject] = ListProperty()
    image: TrapImage = ObjectProperty()
    stats: dict = ObjectProperty()
    bg = ObjectProperty()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        app = App.get_running_app()
        self.classification_threshold = float(
            app.config.get("models", "classification_threshold")
        )

        # Arranging Canvas
        with self.canvas:
            # Update canvas when the window size changes
            self.bind(pos=self.update_rect, size=self.update_rect)

        self.draw()

    # update function which makes the canvas adjustable.
    def update_rect(self, *args):
        if self.bg:
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
                pos_hint={"top": 0},
                keep_ratio=True,
                allow_stretch=True,
            )

        displayed_img_width, displayed_img_height = img.norm_image_size
        source_img_width, source_img_height = img.texture_size
        win_width, win_height = self.size
        x_offset = (win_width / 2) - (displayed_img_width / 2)
        y_offset = (win_height / 2) - (displayed_img_height / 2)
        x_scale = displayed_img_width / source_img_width
        y_scale = displayed_img_height / source_img_height
        self.bg = img
        self.bbox_widgets = []

        # Color(1, 0, 0, 1)
        # frame_thickness = 2
        # self.frame = Line(
        #     rectangle=(
        #         x_offset - frame_thickness,
        #         y_offset - frame_thickness,
        #         displayed_img_width + frame_thickness * 2,
        #         displayed_img_height + frame_thickness * 2,
        #     ),
        #     width=frame_thickness,
        # )

        color = [1, 1, 1, 1]
        for i, annotation in enumerate(self.annotations):
            if not annotation.bbox:
                logger.warn(f"No bbox for detected object {annotation.id}. Skipping.")
                continue

            x1, y1, x2, y2 = annotation.bbox
            w = x2 - x1

            # Reference from bottom left instead of top left
            y1 = source_img_height - y1
            y2 = source_img_height - y2

            # Scale bbox to match on-screen scale of image
            x1 *= x_scale
            y1 *= y_scale
            x2 *= x_scale
            y2 *= y_scale

            w2 = w * x_scale
            w2 = x2 - x1
            x1 += x_offset
            x2 += x_offset
            y1 += y_offset
            y2 += y_offset

            app = App.get_running_app()
            Session = get_session_class(app.db_path)
            with Session() as session:
                best_annotation = annotation.best_sibling(session)
                # track_length = annotation.track_length(session)
                track_info = annotation.track_info(session)

            if best_annotation.binary_label == constants.NEGATIVE_BINARY_LABEL:
                label_text = ""
                color = constants.NEGATIVE_COLOR

            elif (
                best_annotation.specific_label
                and float(best_annotation.specific_label_score)
                >= self.classification_threshold
            ):
                # If there is a binary label and it's nonmoth, don't show
                # the specific label, even if one exists.
                color = constants.POSITIVE_COLOR
                if best_annotation.specific_label_score:
                    score = int(round(best_annotation.specific_label_score * 100, 0))
                    label_text = f"{best_annotation.specific_label} ({score}%)\n"
                else:
                    label_text = f"{best_annotation.specific_label}"

            elif best_annotation.binary_label:
                # Show or not show binary-only labels?
                color = constants.NEUTRAL_COLOR
                # label_text = f"{best_annotation.binary_label.title()}"
                label_text = ""

            else:
                label_text = ""

            if annotation.sequence_id and label_text:
                label_text = (
                    f"{label_text}"
                    f"{annotation.sequence_frame + 1} of {track_info['total_frames']} "
                    f"({format_timedelta_hours(track_info['end_time']-track_info['start_time'])})\n"
                    # f"{annotation.sequence_id}\n"
                    # f"frame {track_info['current_frame']} / {track_info['total_frames']}\n"
                    # f"first seen {track_info['start_time'].strftime('%-I:%-M %p')}\n"
                    # f"last seen {track_info['end_time'].strftime('%-I:%-M %p')}\n"
                )

            # label_text = split_label(label_text)

            with self.canvas:
                Color(*color)

                self.bbox_widgets.append(
                    Line(points=[x1, y1, x1, y2, x2, y2, x2, y1, x1, y1], width=2)
                )

                self.bbox_widgets.append(
                    Label(
                        text=label_text,
                        center=((x1 + w2 / 2), y2 - 42),
                        color=color,
                        bold=True,
                        halign="center",
                    )
                )


class ImagePlaybackScreen(Screen):
    monitoring_session = ObjectProperty()
    image_ids = ListProperty()
    fps = NumericProperty(defaultvalue=DEFAULT_FPS)
    clock = ObjectProperty(allownone=True)

    def reload(self, ms, image_id: Optional[int] = None):
        self.current_sample = None
        app = App.get_running_app()
        self.image_ids = [
            img.id for img in get_monitoring_session_image_ids(app.db_path, ms)
        ]
        preview: PreviewWindow = self.ids.image_preview
        preview.reset()
        if image_id:
            preview.load_sample(image_id)
        else:
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

    def on_enter(self, *args):
        if self.ids.image_preview:
            self.ids.image_preview.start_auto_refresh()

    def on_leave(self, *args):
        self.pause()
        if self.ids.image_preview:
            self.ids.image_preview.stop_auto_refresh()

    def show_summary_screen(self):
        app = App.get_running_app()
        if app:
            screen_name = "summary"
            app.screen_manager.current = screen_name
            if self.monitoring_session:
                app.screen_manager.get_screen(
                    screen_name
                ).monitoring_session = self.monitoring_session


class BBox(BoxLayout):
    pass


class PreviewWindow(RelativeLayout):
    # @TODO save current sample for each directory, so we keep our place
    current_sample = ObjectProperty(allownone=True)
    image_widget = ObjectProperty(allownone=True)
    refresh_clock = ObjectProperty()

    def on_current_sample(self, *args):
        pass

    def reset(self):
        self.current_sample = None
        self.image_widget = None
        self.clear_widgets()

    def on_image_widget(self, *args):
        self.clear_widgets()
        if self.image_widget:
            self.add_widget(self.image_widget)

    def refresh(self, *args):
        self.load_sample(self.current_sample.id)

    def load_sample(self, image_id):
        # Refetch image with associated detected objects
        app = App.get_running_app()
        image = get_image_with_objects(app.db_path, image_id)
        stats = get_object_counts_for_image(app.db_path, image_id)

        # @TODO is there a more reliable way to reference the info bar?
        info_bar = self.parent.parent.ids.info_bar
        update_info_bar(info_bar, image, stats)

        title_bar = self.parent.parent.ids.title_bar
        update_title(title_bar, image)

        image_widget = AnnotatedImage(
            image_path=image.absolute_path,
            annotations=get_unique_objects_for_image(
                db_path=app.db_path, image_id=image_id
            ),
            size=self.size,
            pos_hint={"bottom": 0},
            image=image,
            stats=stats,
        )
        self.current_sample = image
        self.image_widget = image_widget

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

    def add_current_sample_to_queue(self):
        app = App.get_running_app()
        add_image_to_queue(app.db_path, self.current_sample.id)
        self.start_auto_refresh()

    def process_now(self):
        """
        Clear queue then add sample

        @TODO this should skip the queue all together and just process the image in one shot/chain
        """
        app = App.get_running_app()
        clear_all_queues(app.db_path, app.image_base_path)
        delete_objects_for_image(app.db_path, self.current_sample.id)
        add_image_to_queue(app.db_path, self.current_sample.id)
        app.start_queue(single=True)

    def start_auto_refresh(self):
        refresh_interval_seconds = 1

        if self.refresh_clock:
            Clock.unschedule(self.refresh_clock)

        logger.debug(
            f"Scheduling auto-refresh of playback view every {refresh_interval_seconds} seconds"
        )
        self.refresh_clock = Clock.schedule_interval(
            self.refresh, refresh_interval_seconds
        )

    def stop_auto_refresh(self):
        logger.debug("Stopping auto-refresh of playback view")
        if self.refresh_clock:
            Clock.unschedule(self.refresh_clock)

    def button_text(self):
        if self.current_sample and self.current_sample.in_queue:
            return "(In Queue)"
        else:
            return "Process Single"
