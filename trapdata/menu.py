#! /usr/bin/env python3

import asyncio
import time
import pathlib
import random
import tempfile
from xmlrpc.client import Boolean
import dateutil.parser
import logging
import requests
import base64
import io
import threading
from functools import partial

import kivy

kivy.require("2.1.0")


from kivy.app import App
from kivy.config import Config
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.image import Image, AsyncImage
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color, Canvas, Line
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
from kivy.properties import (
    StringProperty,
    ListProperty,
    ObjectProperty,
    NumericProperty,
    BooleanProperty,
)
from kivy.uix.screenmanager import Screen
from kivy.logger import Logger


from .utils import *

from .ml import detect_objects, classify_objects

# detect_and_classify = lambda *args, **kwargs: None


Builder.load_file(str(pathlib.Path(__file__).parent / "menu.kv"))


class ThreadWithStatus(threading.Thread):
    exception = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exception = e
            logger.error(f"Thread {self} exited with an exception: {e}")
            raise e


class TrapSesionData(Widget):
    """
    One night / session of trap data.

    Will keep track of which directories have been processed, their cached results, etc.
    Could be backed by a SQLite database? Or just a folder structure under .cache
    """

    pass


class AnalyzeButton(Button):
    monitoring_session = ObjectProperty()
    images = ListProperty()
    running = BooleanProperty(defaultvalue=False)
    progress = NumericProperty(defaultvalue=0)
    status_label = ObjectProperty()
    progress_label = ObjectProperty()
    popup = ObjectProperty()
    bgtask = ObjectProperty()
    exit_event = ObjectProperty()
    progress_clock = ObjectProperty(allownone=True)

    def on_release(self):
        if not self.popup:
            self.make_popup()
        self.popup.open()

    def on_monitoring_session(self, instance, value):
        ms = value
        self.images = get_monitoring_session_images(ms)

    def start(self, *args):
        # @TODO can use the status property of the bgtask thread instead
        if not self.running:
            self.running = True
            self.progress = 0
            self.exit_event = threading.Event()
            task_name = str(self.monitoring_session)
            self.bgtask = threading.Thread(
                target=self.analyze,
                daemon=True,
                name=task_name,
            )
            self.bgtask.start()
            # self.bgtask.add_done_callback(self.complete)
            self.progress_clock = Clock.schedule_interval(self.increment_progress, 1)

    def stop(self, *args):
        self.running = False
        if self.bgtask:
            self.exit_event.set()

    def increment_progress(self, clk):
        # print(self.bgtask)
        self.progress += 1
        if not self.bgtask.is_alive():
            self.stop()

    def analyze(self):
        # add_sample_to_queue(self.monitoring_session, sample_size=10)

        app = App.get_running_app()

        models_dir = (
            pathlib.Path(app.config.get("models", "user_data_directory")) / "models"
        )

        localization_results_callback = partial(
            save_detected_objects, self.monitoring_session
        )
        detect_objects(
            model_name=app.config.get("models", "localization_model"),
            models_dir=models_dir,
            base_directory=self.monitoring_session.base_directory,
            results_callback=localization_results_callback,
        )

        classification_results_callback = partial(
            save_classified_objects, self.monitoring_session
        )
        classify_objects(
            model_name=app.config.get("models", "binary_classification_model"),
            models_dir=models_dir,
            base_directory=self.monitoring_session.base_directory,
            results_callback=classification_results_callback,
        )

        classification_results_callback = partial(
            save_classified_objects, self.monitoring_session
        )
        classify_objects(
            model_name=app.config.get("models", "taxon_classification_model"),
            models_dir=models_dir,
            base_directory=self.monitoring_session.base_directory,
            results_callback=classification_results_callback,
        )

        self.complete()
        return True

    def complete(self):
        self.running = False
        self.background_color = (0, 1, 0, 1)
        for widget in self.parent.children:
            # @TODO should we register nightly folders by ID somewhere?
            if isinstance(widget, Button):
                widget.disabled = False

    def make_popup(self):
        if not self.popup:
            # @TODO this would be simpler in a kv file now
            content = GridLayout(rows=5, cols=1, spacing=5)
            status = GridLayout(rows=1, cols=2, spacing=0)
            self.status_label = Label(text=f"Running: {self.running}")  # , id="status")
            self.progress_label = Label(text=f"")
            status.add_widget(self.status_label)
            status.add_widget(self.progress_label)
            content.add_widget(status)
            close_button = Button(text="Close", size=(100, 20))
            start_button = Button(text="Start", size=(100, 20))
            start_button.bind(on_press=self.start)
            stop_button = Button(text="Stop", size=(100, 20))
            stop_button.bind(on_press=self.stop)
            content.add_widget(start_button)
            # content.add_widget(stop_button)
            content.add_widget(close_button)
            self.popup = Popup(
                title=f"Detect and classify moths in {str(self.monitoring_session)}",
                content=content,
                auto_dismiss=True,
                size_hint=(None, None),
                size=(600, 600),
            )
            close_button.bind(on_press=self.popup.dismiss)
            self.popup.ids["start_button"] = start_button
            self.popup.ids["stop_button"] = stop_button

    def on_progress(self, instance, value):
        self.progress_label.text = f"{value} seconds"

    def on_running(self, instance, value):
        self.status_label.text = f"Running: {value}"
        if self.popup and value == True:
            self.popup.ids["start_button"].disabled = True
            self.popup.ids["stop_button"].disabled = False
            self.initial_bg_color = self.background_color
            self.background_color = (1, 0, 0, 1)
        else:
            self.popup.ids["start_button"].disabled = False
            self.popup.ids["stop_button"].disabled = True
            self.background_color = getattr(self, "initial_bg_color", None)
            if self.progress_clock:
                Clock.unschedule(self.progress_clock)


class LaunchScreenButton(Button):
    monitoring_session = ObjectProperty()
    screenname = StringProperty(allownone=True)
    screenmanager = ObjectProperty(allownone=True)

    def on_release(self):
        self.launch()

    def launch(self):
        """
        Open the specified screen
        """

        if self.screenmanager and self.screenname:
            self.screenmanager.current = self.screenname
            self.screenmanager.get_screen(
                self.screenname
            ).monitoring_session = self.monitoring_session


class ReportButton(Button):
    monitoring_session = ObjectProperty()

    def on_release(self):
        fname = export_report(self.monitoring_session.base_directory)
        label_text = f"Report saved to: \n{fname}"
        popup = Popup(
            title="Results",
            content=Label(text=label_text),
            size_hint=(None, None),
            size=(f"550dp", 400),
            auto_dismiss=True,
        )
        popup.open()


class DataMenuScreen(Screen):
    root_dir = ObjectProperty(allownone=True)
    sessions = ObjectProperty()
    status_popup = ObjectProperty()
    status_clock = ObjectProperty()
    data_ready = BooleanProperty(defaultvalue=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Clock.schedule_once(self.setup, 1)

    def setup(self, *args):
        if not self.root_dir:
            self.root_dir = choose_directory(cache=True)

    def choose_root_directory(self, *args):
        try:
            self.root_dir = choose_directory(cache=False, starting_path=self.root_dir)
        except Exception as e:
            logger.error("Failed to choose directory with a starting path")
            self.root_dir = choose_directory(cache=False)

    def reload(self):
        """
        Reload the view by changing the root dir.
        """
        root_dir = self.root_dir
        self.root_dir = None
        self.root_dir = root_dir

    def db_ready(self):
        # Try to open a database session. @TODO add GUI indicator and ask to recreate if fails.
        if not db.check_db(self.root_dir):
            Popup(
                title="Error reading database",
                content=Label(
                    text=(
                        f"Error reading database: \n\n"
                        f"{db.db_path(self.root_dir)} \n\n"
                        f"Trying deleting the DB file and it will be recreated on next launch."
                    )
                ),
                size_hint=(None, None),
                size=("550dp", "200dp"),
                # on_dismiss=sys.exit,
            ).open()
            return False
        else:
            return True

    def on_root_dir(self, instance, value):
        root_dir = value
        logger.info("Base directory changed!")
        self.data_ready = False

        if root_dir and self.db_ready():
            label_text = f"Looking for capture data in \n\n{root_dir}"
            self.status_popup = Popup(
                title="Status",
                content=Label(text=label_text),
                size_hint=(None, None),
                size=("550dp", "200dp"),
                auto_dismiss=False,
                on_open=self.get_monitoring_sessions,
            )
            self.status_popup.open()

    def on_data_ready(self, *args):
        if self.data_ready:
            logger.info("Data is ready for other methods")
            # Buttons aren't available immediately
            self.display_monitoring_sessions()
            Clock.schedule_once(self.enable_buttons, 1)
        else:
            self.disable_buttons()

    def enable_buttons(self, *args):
        logger.info("Enabling all buttons")
        for row in self.ids.monitoring_sessions.children:
            for child in row.children:
                if isinstance(child, Button):
                    child.disabled = False

    def disable_buttons(self, *args):
        logger.info("Disabling all buttons")
        for row in self.ids.monitoring_sessions.children:
            for child in row.children:
                if isinstance(child, Button):
                    child.disabled = True

    def get_monitoring_sessions(self, *args):
        self.sessions = get_monitoring_sessions_from_db(self.root_dir)
        if self.sessions:
            self.data_ready = True
        else:
            self.sessions = get_monitoring_sessions_from_filesystem(self.root_dir)
            # @TODO just wait for the DB to save, don't worry about background task
            # Rescan will trigger a scan an resave.
            self.save_monitoring_sessions_in_background()
        self.status_popup.dismiss()

    def save_monitoring_sessions_in_background(self):
        logger.info("Writing monitoring data to DB in the background")
        bgtask = ThreadWithStatus(
            target=partial(save_monitoring_sessions, self.root_dir, self.sessions),
            daemon=True,
            name="writing_monitoring_sessions_to_db",
        )
        bgtask.start()
        self.status_clock = Clock.schedule_interval(
            partial(self.watch_db_progress, bgtask), 1
        )

    def watch_db_progress(self, bgtask, *args):
        logger.debug(f"Checking DB write status: {bgtask}")
        self.ids.status.text = "Writing capture data to the database..."
        if bgtask and not bgtask.is_alive():
            logger.debug(f"Thread has exited: {bgtask}")
            Clock.unschedule(self.status_clock)
            if bgtask.exception:
                self.ids.status.text = "Failed to write capture data to the database"
            else:
                self.sessions = get_monitoring_sessions_from_db(self.root_dir)
                self.data_ready = True
                self.ids.status.text = "Ready"

    def display_monitoring_sessions(self, *args):
        grid = self.ids.monitoring_sessions
        grid.clear_widgets()

        for ms in self.sessions:

            label = (
                f"{ms.day.strftime('%a, %b %e')} \n"
                f"{ms.num_images or 0} images\n"
                f"{ms.duration_label}\n"
                f"{ms.num_detected_objects} objects\n"
            )

            with db.get_session(self.root_dir) as sess:
                first_image = (
                    sess.query(db.Image).filter_by(monitoring_session_id=ms.id).first()
                )

            if first_image:
                first_image_path = pathlib.Path(first_image.path)
                bg_image = str(self.root_dir / first_image_path)
            else:
                continue

            # TEMPORARY until methods use a list of images in DB instead of a path
            path = first_image_path.parent

            # Check if there are unprocessed images in monitoring session?
            btn_disabled = True

            playback_btn = LaunchScreenButton(
                text="Playback",
                monitoring_session=ms,
                screenmanager=self.manager,
                screenname="playback",
                disabled=btn_disabled,
            )

            analyze_btn = AnalyzeButton(
                text="Add to Queue",
                monitoring_session=ms,
                disabled=btn_disabled,
            )

            summary_btn = LaunchScreenButton(
                text="Summary",
                monitoring_session=ms,
                screenmanager=self.manager,
                screenname="summary",
                disabled=btn_disabled,
            )

            row = GridLayout(rows=1, cols=5, spacing=20)
            row.add_widget(AsyncImage(source=bg_image))
            row.add_widget(Label(text=label))
            row.add_widget(playback_btn)
            row.add_widget(analyze_btn)
            row.add_widget(summary_btn)
            grid.add_widget(row)

        self.ids.status.text = "Ready"

    def open_settings(self):
        self.manager.current = "settings"


class DataMenuApp(App):
    def build(self):
        return DataMenuScreen()


if __name__ == "__main__":
    DataMenuApp().run()
