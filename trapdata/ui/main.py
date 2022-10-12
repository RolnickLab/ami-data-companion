# import newrelic.agent
# newrelic.agent.initialize(environment="staging")

import json
import pathlib
import multiprocessing

import kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager
from kivy.uix.settings import SettingsWithSidebar
from kivy.core.window import Window
from kivy.properties import (
    ObjectProperty,
    StringProperty,
    NumericProperty,
    BooleanProperty,
)

from trapdata import logger
from trapdata import ml
from trapdata.db.models.queue import clear_queue

from .menu import DataMenuScreen
from .playback import ImagePlaybackScreen
from .summary import SpeciesSummaryScreen
from .queue import QueueScreen


kivy.require("2.1.0")


# @TODO move this class to the db.models.queue module, it shouldn't be a Kivy object
class Queue(Label):
    app = ObjectProperty()
    status_str = StringProperty(defaultvalue="")
    total_in_queue = NumericProperty(defaultvalue=0)
    clock = ObjectProperty(allownone=True)

    running = BooleanProperty(defaultvalue=False)
    bgtask = ObjectProperty()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("Initializing queue status and starting DB polling")

    def check_queue(self, *args):
        if not self.bgtask:
            self.running = False
        else:
            try:
                self.running = self.bgtask.is_alive()
            except ValueError:
                self.running = False
            else:
                # logger.debug(f"Child process status: {self.bgtask}")
                pass

    def process_queue(self):
        db_path = self.app.db_path

        user_data_path = pathlib.Path(self.app.config.get("paths", "user_data_path"))
        logger.info(f"Local user data path: {user_data_path}")
        num_workers = int(self.app.config.get("performance", "num_workers"))

        model_1_name = self.app.config.get("models", "localization_model")
        Model_1 = ml.models.object_detectors[model_1_name]
        model_1 = Model_1(
            db_path=db_path,
            user_data_path=user_data_path,
            batch_size=int(
                self.app.config.get("performance", "localization_batch_size")
            ),
            num_workers=num_workers,
        )
        model_1.run()
        logger.info("Localization complete")

        model_2_name = self.app.config.get("models", "binary_classification_model")
        Model_2 = ml.models.binary_classifiers[model_2_name]
        model_2 = Model_2(
            db_path=db_path,
            user_data_path=user_data_path,
            batch_size=int(
                self.app.config.get("performance", "classification_batch_size")
            ),
            num_workers=num_workers,
        )
        model_2.run()
        logger.info("Binary classification complete")

        model_3_name = self.app.config.get("models", "taxon_classification_model")
        Model_3 = ml.models.species_classifiers[model_3_name]
        model_3 = Model_3(
            db_path=db_path,
            user_data_path=user_data_path,
            batch_size=int(
                self.app.config.get("performance", "classification_batch_size")
            ),
            num_workers=num_workers,
        )
        model_3.run()
        logger.info("Species classification complete")

    def on_running(self, *args):
        if self.running:
            if not self.clock:
                logger.debug("Scheduling queue check")
                self.clock = Clock.schedule_interval(self.check_queue, 1)
            self.status_str = "Running"
        else:
            # logger.debug("Unscheduling queue check")
            # Clock.unschedule(self.clock)
            self.status_str = "Stopped"

    def start(self, *args):
        # @NOTE can't change a widget property from a bg thread
        if not self.running:
            if self.bgtask:
                self.cleanup()
            logger.info("Starting queue")
            task_name = "Trapdata Queue Processor"
            self.bgtask = multiprocessing.Process(
                target=self.process_queue,
                daemon=False,
                name=task_name,
            )
            logger.info(f"Starting child process {self.bgtask}")
            self.bgtask.start()
            logger.info(f"Started child process {self.bgtask}")
            self.running = True

    def stop(self, *args):
        if self.bgtask:
            # self.bgtask.terminate()
            logger.info(f"Killing child process {self.bgtask}")
            self.bgtask.kill()
            logger.info(f"Waiting for {self.bgtask}")
            self.bgtask.join()

    def cleanup(self, *args):
        if self.bgtask:
            try:
                logger.info(f"Cleaning up {self.bgtask}")
                self.bgtask.close()
            except ValueError:
                pass

    def clear(self):
        clear_queue(self.app.db_path)


class TrapDataApp(App):
    # @TODO this db_session is not currently used, but may be more
    # convenient that the current usage of DB sessions.
    queue = ObjectProperty()
    image_base_path = StringProperty(allownone=True)
    screen_manager = ObjectProperty()
    use_kivy_settings = False

    def on_stop(self):
        # The Kivy event loop is about to stop, set a stop signal;
        # otherwise the app window will close, but the Python process will
        # keep running until all secondary threads exit.
        # @TODO Set stop byte in the database
        # @TODO stop background threads
        for child in multiprocessing.active_children():
            try:
                logger.info(f"Killing child process {child}")
                child.kill()
                logger.info(f"Waiting for child process {child}")
                child.join()
                logger.info(f"Clean up child process {child}")
                child.close()
            except ValueError:
                pass

    @property
    def db_path(self):
        return self.config.get("paths", "database_url")

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
        sm.add_widget(QueueScreen(name="queue"))
        self.screen_manager = sm

        return sm

    def on_image_base_path(self, *args):
        """
        When a base path is set, create a queue status
        """
        if self.queue and self.queue.clock:
            Clock.unschedule(self.queue.clock)
        self.queue = Queue(app=self)

    def start_queue(self):
        if self.queue:
            self.queue.start()
        else:
            logger.warn("No queue found!")

    def stop_queue(self):
        if self.queue:
            self.queue.stop()
        else:
            logger.warn("No queue found!")

    def build_config(self, config):
        default_db_connection_string = (
            f"sqlite+pysqlite:///{pathlib.Path(self.user_data_dir) / 'trapdata.db'}"
        )
        config.setdefaults(
            "paths",
            {
                "user_data_path": self.user_data_dir,
                # "image_base_path": self.user_data_dir,
                "database_url": default_db_connection_string,
            },
        )
        config.setdefaults(
            "models",
            {
                "localization_model": list(ml.models.object_detectors.keys())[0],
                "binary_classification_model": list(
                    ml.models.binary_classifiers.keys()
                )[0],
                "taxon_classification_model": list(
                    ml.models.species_classifiers.keys()
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
                "num_workers": 1,
            },
        )
        # config.write()

    def build_settings(self, settings):
        path_settings = [
            {
                "key": "user_data_path",
                "type": "path",
                "title": "Local directory for models, thumbnails & reports",
                "desc": "Model weights are between 100-200Mb and will be downloaded the first time a model is used.",
                "section": "paths",
            },
            {
                "key": "database_url",
                "type": "string",
                "title": "Database connection string",
                "desc": "Defaults to a local SQLite database that will automatically be created. Supports PostgreSQL.",
                "section": "paths",
            },
        ]
        model_settings = [
            {
                "key": "localization_model",
                "type": "options",
                "title": "Localization model",
                "desc": "Model & settings to use for object detection in original images from camera trap.",
                "options": list(ml.models.object_detectors.keys()),
                "section": "models",
            },
            {
                "key": "binary_classification_model",
                "type": "options",
                "title": "Binary classification model",
                "desc": "Model & settings to use for moth / non-moth classification of cropped images after object detection.",
                "options": list(ml.models.binary_classifiers.keys()),
                "section": "models",
            },
            {
                "key": "taxon_classification_model",
                "type": "options",
                "title": "Species classification model",
                "desc": "Model & settings to use for fine-grained species or taxon-level classification of cropped images after moth/non-moth detection.",
                "options": list(ml.models.species_classifiers.keys()),
                "section": "models",
            },
            {
                "key": "tracking_algorithm",
                "type": "options",
                "title": "Occurence tracking algorithm (de-duplication)",
                "desc": "Method of identifying and tracking the same individual moth across multiple images.",
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
            "Paths",
            self.config,
            data=json.dumps(path_settings),
        )
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


# @newrelic.agent.background_task()
def run():
    TrapDataApp().run()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(TrapDataAnalyzer().async_run())
    # loop.close()
