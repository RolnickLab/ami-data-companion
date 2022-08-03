import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen

from .menu import DataMenuScreen
from .playback import ImagePlaybackScreen
from .summary import SpeciesSummaryScreen


kivy.require("2.1.0")


class TrapDataAnalyzer(App):
    def build(self):
        self.title = "AMI Trap Data Companion"

        sm = ScreenManager()
        sm.add_widget(DataMenuScreen(name="menu"))
        sm.add_widget(ImagePlaybackScreen(name="playback"))
        sm.add_widget(SpeciesSummaryScreen(name="summary"))
        return sm


def run():
    TrapDataAnalyzer().run()
