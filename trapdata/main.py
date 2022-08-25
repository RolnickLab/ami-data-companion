# import asyncio

import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty

from .menu import DataMenuScreen
from .playback import ImagePlaybackScreen
from .summary import SpeciesSummaryScreen
from .db import get_session


kivy.require("2.1.0")


class TrapDataAnalyzer(App):
    db_session = ObjectProperty()

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

        sm = ScreenManager()
        sm.add_widget(DataMenuScreen(name="menu"))
        sm.add_widget(ImagePlaybackScreen(name="playback"))
        sm.add_widget(SpeciesSummaryScreen(name="summary"))

        return sm


def run():
    TrapDataAnalyzer().run()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(TrapDataAnalyzer().async_run())
    # loop.close()
