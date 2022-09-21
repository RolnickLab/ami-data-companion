# import asyncio
import json

import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.settings import SettingsWithSidebar
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.config import Config

from .menu import DataMenuScreen
from .playback import ImagePlaybackScreen
from .summary import SpeciesSummaryScreen
from .db import get_session


kivy.require("2.1.0")


class TrapDataAnalyzer(App):
    # @TODO this db_session is not currently used, but may be more
    # convenient that the current usage of DB sessions.
    db_session = ObjectProperty()
    use_kivy_settings = False

    def on_stop(self):
        # The Kivy event loop is about to stop, set a stop signal;
        # otherwise the app window will close, but the Python process will
        # keep running until all secondary threads exit.
        if hasattr(self.root, "stop"):
            self.root.stop.set()
        if self.db_session:
            self.db_session.close()

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

    def build_config(self, config):
        config.setdefaults(
            "models",
            {
                "booltest": True,
                "localization_model": None,
                "binary_classification_model": None,
                "taxon_classification_model": None,
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
                "key": "localization_model",
                "type": "options",
                "title": "Localization model",
                "desc": "Model to use for object detection in original images from camera trap.",
                "options": ["model1", "model2", "model3"],
                "section": "models",
            },
            {
                "key": "binary_classification_model",
                "type": "options",
                "title": "Binary classification model",
                "desc": "Model to use for moth / non-moth classification of cropped images after object detection.",
                "options": ["model1", "model2", "model3"],
                "section": "models",
            },
            {
                "key": "taxon_classification_model",
                "type": "options",
                "title": "Species classification model",
                "desc": "Model to use for fine-grained species or taxon-level classification of cropped images after moth/non-moth detection.",
                "options": ["model1", "model2", "model3"],
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
