import pathlib
import time
from functools import partial

from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty, ListProperty, NumericProperty
from kivy.uix.label import Label
from kivy.uix.button import ButtonBehavior, Button
from kivy.uix.recycleview import RecycleView
from kivy.uix.boxlayout import BoxLayout

# from kivy.uix.carousel
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.clock import Clock

from trapdata import logger
from trapdata import constants
from trapdata.ui.playback import ImagePlaybackScreen
from trapdata.db.models.events import MonitoringSession
from trapdata.db.models.detections import (
    get_detected_objects,
    export_detected_objects,
    get_unique_species_by_track,
)


Builder.load_file(str(pathlib.Path(__file__).parent / "summary.kv"))

NUM_EXAMPLES_PER_ROW = 1


def load_image_in_playback(monitoring_session: MonitoringSession, image_id: int):
    app = App.get_running_app()
    if app:
        screen_name = "playback"
        app.screen_manager.current = screen_name
        playback: ImagePlaybackScreen = app.screen_manager.get_screen(screen_name)
        playback.reload(ms=monitoring_session, image_id=image_id)


class ImageToPlaybackButton(ButtonBehavior, Image):
    monitoring_session = ObjectProperty()
    source_image_id = NumericProperty()

    def on_release(self):
        if self.monitoring_session and self.source_image_id:
            load_image_in_playback(self.monitoring_session, self.source_image_id)


class SpeciesRow(BoxLayout):
    species = ObjectProperty(allownone=True)
    heading = ListProperty(allownone=True)
    image_column_size_hint_x = 2.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_species(self, instance, value):
        if self.species:
            self.make_row(self.species)

    def on_heading(self, instance, value):
        if self.heading:
            self.make_heading(self.heading)

    def make_heading(self, heading: list[str]):
        self.clear_widgets()
        for i, value in enumerate(heading):
            label = Label(
                text=value,
                halign="center",
                valign="top",
                bold=True,
            )
            if i == 0:
                # This is the image thumbnail
                label.size_hint_x = self.image_column_size_hint_x
            label.bind(size=label.setter("text_size"))
            self.add_widget(label)

    def make_row(self, species: dict):
        self.clear_widgets()

        label_text = f"{species['name']}\n\n{species['sequence']}"

        label = Label(
            text=label_text,
            halign="left",
            valign="middle",
        )
        label.bind(size=label.setter("text_size"))

        cell = BoxLayout(padding=(20, 0), size_hint_x=self.image_column_size_hint_x)
        for i in range(NUM_EXAMPLES_PER_ROW):
            try:
                example = species["examples"][i]
                image_widget = ImageToPlaybackButton(
                    source=example["cropped_image_path"],
                    size_hint_y=None,
                    height=species["image_height"],
                    monitoring_session=species["monitoring_session"],
                    source_image_id=example["source_image_id"],
                )
            except IndexError:
                image_widget = Label(text="")

            cell.add_widget(image_widget)

        self.add_widget(cell)
        self.add_widget(label)
        self.add_widget(
            Label(text=str(species["count"]), valign="middle", halign="center")
        )
        self.add_widget(
            Label(
                text=str(round(species["score"] * 100, 1)),
                valign="middle",
                halign="center",
            )
        )
        self.add_widget(
            Label(text=str(species["start_time"]), valign="middle", halign="center")
        )
        self.add_widget(
            Label(text=str(species["duration"]), valign="middle", halign="center")
        )


class SpeciesListLayout(RecycleView):
    monitoring_session = ObjectProperty()
    refresh_clock = ObjectProperty(allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []

    def on_monitoring_session(self, instance, value):
        self.refresh()
        self.start_auto_refresh()

    def start_auto_refresh(self):
        # refresh_interval_seconds = constants.SUMMARY_REFRESH_SECONDS
        refresh_interval_seconds = (
            60 * 5
        )  # @TODO refresh is getting slow, freezes the interface and bumps you to the top

        if self.refresh_clock:
            Clock.unschedule(self.refresh_clock)

        logger.debug(
            f"Scheduling auto-refresh of species list every {refresh_interval_seconds} seconds"
        )
        self.refresh_clock = Clock.schedule_interval(
            self.refresh, refresh_interval_seconds
        )

    def stop_auto_refresh(self):
        logger.debug("Stopping auto-refresh of species list")
        if self.refresh_clock:
            Clock.unschedule(self.refresh_clock)

    def refresh(self, *args):
        # logger.debug("Refreshing species list")
        self.data = self.load_species(self.monitoring_session)
        self.refresh_from_data()

    def load_species(self, ms):
        """
        Return a list of species in the format that the "viewclass" widget
        expects. In this case the viewclass is a `SpeciesRow` object.
        """
        app = App.get_running_app()
        classification_threshold = float(
            app.config.get("models", "classification_threshold")
        )
        # classification_summary = queries.summarize_results(
        #     app.db_path,
        #     ms,
        #     classification_threshold=classification_threshold,
        #     num_examples=NUM_EXAMPLES_PER_ROW,
        # )
        classification_summary = get_unique_species_by_track(
            app.db_path,
            ms,
            classification_threshold=classification_threshold,
            num_examples=NUM_EXAMPLES_PER_ROW,
        )

        row_height = 300  # @TODO make dynamic? Or fit images to specific size

        species_rows = [
            {
                "species": {
                    "sequence": item["sequence_id"].split("-", 1)[-1],
                    "name": item["label"] or "Unclassified",
                    "count": item["sequence_frame_count"],
                    "score": item["sequence_best_score"],
                    "examples": item["examples"],
                    "start_time": item["sequence_start_time"].strftime("%H:%m"),
                    "duration": item["sequence_duration"],
                    "image_height": row_height,
                    "monitoring_session": self.monitoring_session,
                },
                "heading": None,
                "height": row_height,
            }
            for item in classification_summary
        ]

        example_placeholders = [""] * (NUM_EXAMPLES_PER_ROW - 1)
        header_row = [
            {
                "species": None,
                "heading": [
                    "Example",
                    "Label",
                    "Frames",
                    "Score",
                    "Arrival",
                    "Duration",
                ]
                + example_placeholders,
                "height": 50,
            }
        ]

        return header_row + species_rows


class SpeciesSummaryScreen(Screen):
    monitoring_session = ObjectProperty()

    def on_monitoring_session(self, instance, value):
        """
        Update the monitoring session of the child SpeciesListLayout
        when the monitoring session of this screen changes.
        """
        self.ids.species_list.monitoring_session = self.monitoring_session

    def exit(self):
        self.manager.current = "menu"

    def refresh(self):
        self.ids.species_list.refresh()

    def on_enter(self, *args):
        self.ids.species_list.start_auto_refresh()

    def on_leave(self, *args):
        self.ids.species_list.stop_auto_refresh()

    def export(self):
        app = App.get_running_app()
        if app:
            records = list(
                get_detected_objects(
                    db_path=app.db_path,
                    image_base_path=app.image_base_path,
                    monitoring_session=self.monitoring_session,
                )
            )
            timestamp = int(time.time())
            trap = pathlib.Path(app.image_base_path).name
            report_name = f"{trap}-detections-for-{self.monitoring_session.day.strftime('%Y-%m-%d')}-created-{timestamp}"
            app.export_detections(detected_objects=records, report_name=report_name)
