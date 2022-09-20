from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image, AsyncImage
from kivy.lang import Builder
from kivy.clock import Clock

from .utils import *

from . import db


Builder.load_file(str(pathlib.Path(__file__).parent / "settings.kv"))


class SettingsScreen(Screen):
    def exit(self):
        self.manager.current = "menu"
