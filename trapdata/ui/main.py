# import newrelic.agent
# newrelic.agent.initialize(environment="staging")

import json
import pathlib

# import multiprocessing
import threading
import time
from functools import partial
from typing import Any, Dict, Optional, cast

import kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.config import ConfigParser
from kivy.core.window import Window
from kivy.properties import (
    BooleanProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
)
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import NoTransition, ScreenManager
from kivy.uix.settings import SettingsWithSidebar
from rich import print
from sqlalchemy import orm

from trapdata import logger, ml
from trapdata.db.models.detections import export_detected_objects, get_detected_objects
from trapdata.db.models.events import (
    export_monitoring_sessions,
    get_monitoring_sessions_from_db,
)
from trapdata.db.models.queue import clear_all_queues
from trapdata.settings import Settings
from trapdata.ui.pipeline import start_pipeline

from .menu import DataMenuScreen
from .playback import ImagePlaybackScreen
from .queue import QueueScreen
from .species_summary import SpeciesSummaryGridScreen
from .summary import SpeciesSummaryScreen

kivy.require("2.1.0")


# @TODO move this class to the db.models.queue module, it shouldn't be a Kivy object
class Queue(Label):
    app = ObjectProperty()
    status_str = StringProperty(defaultvalue="")
    total_in_queue = NumericProperty(defaultvalue=0)
    clock = ObjectProperty(allownone=True)

    running = BooleanProperty(defaultvalue=False)
    bgtask = ObjectProperty(allownone=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("Initializing queue status and starting DB polling")

    def check_queue(self, *args):
        if not self.bgtask:
            self.running = False
        else:
            try:
                new_status = self.bgtask.is_alive()
                if self.running and new_status is False:
                    # Pipeline went from running to stopped, reload stats on menu screen
                    menu = self.app.screen_manager.get_screen("menu")
                    if menu:
                        menu.reload()
                self.running = self.bgtask.is_alive()
            except ValueError:
                self.running = False
                self.bgtask = None
            else:
                # logger.debug(f"Child process status: {self.bgtask}")
                pass

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
        if self.bgtask:
            logger.info(f"Background task changed status: {self.bgtask}")

    def start(self, *args, single: bool = False):
        # @NOTE can't change a widget property from a bg thread
        if not self.running:
            if self.bgtask:
                self.cleanup()
            logger.info("Starting queue")
            task_name = "Trapdata Queue Processor"
            self.bgtask = threading.Thread(
                target=partial(
                    start_pipeline,
                    db_path=self.app.db_path,
                    image_base_path=self.app.image_base_path,
                    config=self.app.config,
                    single=single,
                ),
                daemon=True,  # PyTorch will be killed abruptly, leaving memory in GPU
                name=task_name,
            )
            logger.info(f"Starting child process {self.bgtask}")
            self.bgtask.start()
            logger.info(f"Started child process {self.bgtask}")
            self.running = True

    def stop(self, *args):
        logger.warn(
            "Stop not implemented. It's not possible to kill a background thread"
        )
        # The following code is for background Processes
        # if self.bgtask:
        #    try:
        #        logger.info(f"Killing child process {self.bgtask}")
        #        # self.bgtask.terminate()
        #        self.bgtask.kill()
        #        logger.info(f"Waiting for {self.bgtask}")
        #        self.bgtask.join()
        #    except ValueError:
        #        self.bgtask = None
        #    else:
        #        self.cleanup()

    def cleanup(self, *args):
        logger.debug("No cleanup implemented for background thread")
        # The following code is for background Processes
        # if self.bgtask:
        #     try:
        #         logger.info(f"Cleaning up {self.bgtask}")
        #         self.bgtask.close()
        #     except ValueError:
        #         pass
        #     finally:
        #         self.bgtask = None

    def clear(self):
        self.stop()
        clear_all_queues(self.app.db_path, self.app.image_base_path)


class TrapDataApp(App):
    # @TODO this db_session is not currently used, but may be more
    # convenient that the current usage of DB sessions.
    queue = ObjectProperty()
    image_base_path = StringProperty(allownone=True)
    screen_manager = ObjectProperty()
    use_kivy_settings = False
    app_settings = ObjectProperty()
    config: Optional[ConfigParser] = ObjectProperty()

    def on_stop(self):
        # The Kivy event loop is about to stop, set a stop signal;
        # otherwise the app window will close, but the Python process will
        # keep running until all secondary threads exit.
        # @TODO Set stop byte in the database
        # @TODO stop background threads
        # Disabling multiprocessing now, due to cross-platform issues
        # for child in multiprocessing.active_children():
        #     try:
        #         logger.info(f"Killing child process {child}")
        #         child.kill()
        #         logger.info(f"Waiting for child process {child}")
        #         child.join()
        #         logger.info(f"Clean up child process {child}")
        #         child.close()
        #     except ValueError:
        #         pass
        orm.close_all_sessions()

    @property
    def db_path(self) -> Optional[str]:
        if self.config:
            return self.config.get("paths", "database_url")
        return None

    def build(self):
        self.title = "AMI Trap Data Companion"
        self.settings_cls = SettingsWithSidebar

        # Just in case we are in a bind:
        Window.fullscreen = 0
        Window.show_cursor = True
        # Window.clearcolor = (1, 1, 1, 1.0)
        # Window.size = (600, 400)

        sm = ScreenManager(transition=NoTransition())
        sm.add_widget(DataMenuScreen(name="menu"))
        sm.add_widget(ImagePlaybackScreen(name="playback"))
        sm.add_widget(SpeciesSummaryScreen(name="summary"))
        sm.add_widget(SpeciesSummaryGridScreen(name="species_grid"))
        sm.add_widget(QueueScreen(name="queue"))
        self.screen_manager = sm

        self.refresh_app_settings()

        return sm

    def refresh_app_settings(self):
        """
        Create a Pydantic BaseSettings instance for accessing
        the app settings in a standardized way whether functions are called
        from the GUI, from the CLI or another API.

        The Settings class reads the Kivy settings file.
        """
        self.app_settings = Settings()  # Initialize without env file
        print(self.app_settings)

    def on_config_change(self, config, section, key, value):
        if key == "image_base_path":
            self.image_base_path = value
        self.refresh_app_settings()
        return super().on_config_change(config, section, key, value)

    def on_image_base_path(self, *args):
        """
        When a base path is set, create a queue status
        """
        if (
            self.screen_manager
            and self.screen_manager.current == "menu"
            and self.screen_manager.current_screen
        ):
            current_screen = self.screen_manager.current_screen
            if hasattr(current_screen, "image_base_path"):
                current_screen.image_base_path = self.image_base_path

        if self.queue and self.queue.clock:
            Clock.unschedule(self.queue.clock)
        self.queue = Queue(app=self)

    def start_queue(self, single=False):
        if self.queue:
            self.queue.start(single=single)
        else:
            logger.warn("No queue found!")

    def stop_queue(self):
        if self.queue:
            self.queue.stop()
        else:
            logger.warn("No queue found!")

    def clear_queue(self):
        if self.queue:
            self.queue.clear()
        else:
            logger.warn("No queue found!")

    def build_config(self, config):
        # @TODO get these defaults from settings.py
        default_db_connection_string = (
            f"sqlite+pysqlite:///{pathlib.Path(self.user_data_dir) / 'trapdata.db'}"
        )
        config.setdefaults(
            "paths",
            {
                "image_base_path": "",  # Using None here gets converted into a string
                "user_data_path": self.user_data_dir,
                "database_url": default_db_connection_string,
            },
        )
        config.setdefaults(
            "models",
            {
                "localization_model": ml.models.DEFAULT_OBJECT_DETECTOR,
                "binary_classification_model": ml.models.DEFAULT_BINARY_CLASSIFIER,
                "species_classification_model": ml.models.DEFAULT_SPECIES_CLASSIFIER,
                "feature_extractor": ml.models.DEFAULT_FEATURE_EXTRACTOR,
                "classification_threshold": 0.6,
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
        self.config = config

    def build_settings(self, settings):
        kivy_settings = {}
        schema = cast(Dict[str, Any], Settings.model_json_schema())
        properties = cast(Dict[str, Any], schema.get("properties", {}))

        # Get fields info from model config
        model_config = cast(Dict[str, Any], Settings.model_config)
        json_schema_extra = cast(
            Dict[str, Any], model_config.get("json_schema_extra", {})
        )
        fields_info = cast(Dict[str, Any], json_schema_extra.get("fields", {}))

        for key, options in properties.items():
            field_info = fields_info.get(key, {})
            section = field_info.get("kivy_section", "Other")
            type_ = field_info.get("kivy_type", "string")

            kivy_settings.setdefault(section, [])
            setting = {
                "key": key,
                "type": type_,
                "title": field_info.get("title", key),
                "desc": field_info.get("description", ""),
                "section": section,
            }

            # Handle enum options
            if type_ == "options" and isinstance(options, dict) and "enum" in options:
                setting["options"] = options["enum"]

            kivy_settings[section].append(setting)

        if self.config:
            for section, items in kivy_settings.items():
                settings.add_json_panel(
                    section.title(),
                    self.config,
                    data=json.dumps(items, default=str),
                )
            if hasattr(self.config, "filename"):
                logger.info(f"Kivy settings file: {self.config.filename}")

    def export_events(self):
        """
        User initiated export of Monitoring Sessions / Survey Events
        with a pop-up.
        """
        if not self.config:
            logger.error("No config available")
            return

        user_data_path = self.config.get("paths", "user_data_path")
        if not self.db_path:
            logger.error("No database path available")
            return

        items = list(
            get_monitoring_sessions_from_db(
                db_path=self.db_path,
                base_directory=self.image_base_path or "",
            )
        )
        timestamp = int(time.time())
        trap = (
            pathlib.Path(self.image_base_path).name
            if self.image_base_path
            else "unknown"
        )
        report_name = f"{trap}-monitoring_events-{timestamp}"
        filepath = export_monitoring_sessions(
            items=items,
            directory=user_data_path,
            report_name=report_name,
        )
        if filepath:
            logger.info(f"Exported monitoring events to {filepath}")
            msg = (
                f"{len(items)} events have been exported to: \n\n"
                f'"{filepath.name}" \n\n'
                f"In the directory: \n{filepath.parent} \n"
            )
        else:
            msg = "Nothing exported, no report created"
            logger.warn(msg)
        Popup(
            title="Report exported",
            content=Label(text=msg),
            size_hint=(None, None),
            size=("550dp", "220dp"),
        ).open()

    def export_detections(self, detected_objects=None, report_name=None):
        """
        User initiated export of Detected Objects with a pop-up.
        """
        if not self.config:
            logger.error("No config available")
            return

        user_data_path = self.config.get("paths", "user_data_path")
        if not self.db_path:
            logger.error("No database path available")
            return

        objects = list(
            detected_objects
            or get_detected_objects(
                db_path=self.db_path,
                image_base_path=(
                    self.config.get("paths", "image_base_path") if self.config else ""
                ),
                classification_threshold=(
                    float(self.config.get("models", "classification_threshold"))
                    if self.config
                    else 0.6
                ),
            )
        )
        timestamp = int(time.time())
        trap = (
            pathlib.Path(self.image_base_path).name
            if self.image_base_path
            else "unknown"
        )
        report_name = report_name or f"{trap}-all-detections-{timestamp}"
        filepath = export_detected_objects(
            items=objects,
            directory=user_data_path,
            report_name=report_name,
        )
        if filepath:
            logger.info(f"Exported detections to {filepath}")
            msg = (
                f"{len(objects)} detected objects have been exported to: \n\n"
                f'"{filepath.name}" \n\n'
                f"In the directory: \n{filepath.parent} \n"
            )
        else:
            msg = "Nothing exported, no report created"
            logger.warn(msg)
        Popup(
            title="Report exported",
            content=Label(text=msg),
            size_hint=(None, None),
            size=("550dp", "220dp"),
        ).open()


# @newrelic.agent.background_task()
def run():
    TrapDataApp().run()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(TrapDataAnalyzer().async_run())
    # loop.close()
