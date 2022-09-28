# import asyncio
import json
import pathlib
from functools import partial
import threading

import kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.settings import SettingsWithSidebar
from kivy.core.window import Window
from kivy.properties import (
    ObjectProperty,
    StringProperty,
    NumericProperty,
    BooleanProperty,
)
from kivy.config import Config

from .menu import DataMenuScreen
from .playback import ImagePlaybackScreen
from .summary import SpeciesSummaryScreen
from . import ml
from .utils import *


kivy.require("2.1.0")


class Queue(Label):
    app = ObjectProperty()
    status_str = StringProperty(defaultvalue="")
    total_in_queue = NumericProperty(defaultvalue=0)
    clock = ObjectProperty(allownone=True)

    running = BooleanProperty(defaultvalue=False)
    bgtask = ObjectProperty()

    exit_event = ObjectProperty()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("Initializing queue status and starting DB polling")

    def on_total_in_queue(self, *args):
        msg = f"{self.total_in_queue} images in queue"
        logger.debug(msg)
        self.status_str = msg

    def check_queue(self, *args):
        self.running = self.bgtask.is_alive()
        logger.debug(f"Checking queue, running: {self.running}")
        logger.info(queue_counts(self.app.base_path))
        self.total_in_queue = images_in_queue(self.app.base_path)

    def process_queue(self):
        base_path = self.app.base_path

        models_dir = (
            pathlib.Path(self.app.config.get("models", "user_data_directory"))
            / "models"
        )

        localization_results_callback = partial(save_detected_objects, base_path)
        ml.detect_objects(
            model_name=self.app.config.get("models", "localization_model"),
            models_dir=models_dir,
            base_directory=base_path,  # base path for relative images
            results_callback=localization_results_callback,
        )
        logger.info("Localization complete")

        classification_results_callback = partial(save_classified_objects, base_path)
        ml.classify_objects(
            model_name=self.app.config.get("models", "binary_classification_model"),
            models_dir=models_dir,
            base_directory=base_path,
            results_callback=classification_results_callback,
        )
        logger.info("Binary classification complete")

        classification_results_callback = partial(save_classified_objects, base_path)
        ml.classify_objects(
            model_name=self.app.config.get("models", "taxon_classification_model"),
            models_dir=models_dir,
            base_directory=base_path,
            results_callback=classification_results_callback,
        )
        logger.info("Species classification complete")

    def on_running(self, *args):
        if self.running:
            if not self.clock:
                logger.debug("Scheduling queue check")
                self.clock = Clock.schedule_interval(self.check_queue, 1)
        else:
            logger.debug("Unscheduling queue check")
            Clock.unschedule(self.clock)

    def start(self, *args):
        # @NOTE can't change a widget property from a bg thread
        logger.info("Starting queue")
        if not self.running:
            self.running = True
            task_name = "Mr. Queue"
            self.bgtask = threading.Thread(
                target=self.process_queue,
                daemon=True,
                name=task_name,
            )
            self.bgtask.start()

    def clear(self):
        clear_queue(self.app.base_path)


class TrapDataAnalyzer(App):
    # @TODO this db_session is not currently used, but may be more
    # convenient that the current usage of DB sessions.
    queue = ObjectProperty()
    base_path = StringProperty(allownone=True)
    use_kivy_settings = False

    def on_stop(self):
        # The Kivy event loop is about to stop, set a stop signal;
        # otherwise the app window will close, but the Python process will
        # keep running until all secondary threads exit.
        # @TODO Set stop byte in the database
        # @TODO stop background threads
        pass

    def build(self):
        self.title = "AMI Trap Data Companion"
        self.settings_cls = SettingsWithSidebar

        # Just in case we are in a bind:
        Window.fullscreen = 0
        Window.show_cursor = True

        sm = ScreenManager()
        sm.add_widget(DataMenuScreen(name="menu"))
        sm.add_widget(ImagePlaybackScreen(name="playback"))
        sm.add_widget(SpeciesSummaryScreen(name="summary"))

        return sm

    def on_base_path(self, *args):
        """
        When a DB path is set, create a queue status
        """
        if self.queue and self.queue.clock:
            Clock.unschedule(self.queue.clock)
        self.queue = Queue(app=self)

    def start_queue(self):
        if self.queue:
            self.queue.start()
        else:
            logger.warn("No queue found!")

    def build_config(self, config):
        config.setdefaults(
            "models",
            {
                "user_data_directory": self.user_data_dir,
                "localization_model": list(ml.LOCALIZATION_MODELS.keys())[0],
                "binary_classification_model": list(
                    ml.BINARY_CLASSIFICATION_MODELS.keys()
                )[0],
                "taxon_classification_model": list(
                    ml.TAXON_CLASSIFICATION_MODELS.keys()
                )[0],
                "tracking_algorithm": None,
            },
        )
        config.setdefaults(
            "performance",
            {
                "use_gpu": 1,
                "localization_batch_size": 2,
                "classification_batch_size": 20,
                "num_workers": 4,
            },
        )
        # config.write()

    def build_settings(self, settings):
        model_settings = [
            {
                "key": "user_data_directory",
                "type": "path",
                "title": "Local directory for model data",
                "desc": "Model weights are between 100-200Mb and will be downloaded the first time a model is used.",
                "section": "models",
            },
            {
                "key": "localization_model",
                "type": "options",
                "title": "Localization model",
                "desc": "Model & settings to use for object detection in original images from camera trap.",
                "options": list(ml.LOCALIZATION_MODELS.keys()),
                "section": "models",
            },
            {
                "key": "binary_classification_model",
                "type": "options",
                "title": "Binary classification model",
                "desc": "Model & settings to use for moth / non-moth classification of cropped images after object detection.",
                "options": list(ml.BINARY_CLASSIFICATION_MODELS.keys()),
                "section": "models",
            },
            {
                "key": "taxon_classification_model",
                "type": "options",
                "title": "Species classification model",
                "desc": "Model & settings to use for fine-grained species or taxon-level classification of cropped images after moth/non-moth detection.",
                "options": list(ml.TAXON_CLASSIFICATION_MODELS.keys()),
                "section": "models",
            },
            {
                "key": "tracking_algorithm",
                "type": "options",
                "title": "Occurence tracking algorithm (de-duplication)",
                "desc": "Method of identifying and tracking the same individual moth accross multiple images.",
                "options": [],
                "section": "models",
            },
        ]

        performance_settings = [
            {
                "key": "use_gpu",
                "type": "bool",
                "title": "Use GPU if available",
                "section": "performance",
            },
            {
                "key": "localization_batch_size",
                "type": "numeric",
                "title": "Localization batch size",
                "desc": (
                    "Number of images to process per-batch during localization. "
                    "These are large images (e.g. 4096x2160px), smaller batch sizes are appropriate (1-10). "
                    "Reduce this if you run out of memory."
                ),
                "section": "performance",
            },
            {
                "key": "classification_batch_size",
                "type": "numeric",
                "title": "Classification batch size",
                "desc": (
                    "Number of images to process per-batch during classification. "
                    "These are small images (e.g. 50x100px), larger batch sizes are appropriate (10-200). "
                    "Reduce this if you run out of memory."
                ),
                "section": "performance",
            },
            {
                "key": "num_workers",
                "type": "numeric",
                "title": "Number of workers",
                "desc": "Number of parallel workers for the PyTorch dataloader. See https://pytorch.org/docs/stable/data.html",
                "section": "performance",
            },
        ]

        settings.add_json_panel(
            "Model selection",
            self.config,
            data=json.dumps(model_settings),
        )
        settings.add_json_panel(
            "Performance settings",
            self.config,
            data=json.dumps(performance_settings),
        )


def run():
    TrapDataAnalyzer().run()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(TrapDataAnalyzer().async_run())
    # loop.close()
