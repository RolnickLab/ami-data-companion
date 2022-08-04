#! /usr/bin/env python3

import pathlib
import random
import tempfile
import dateutil.parser
import logging
import requests
import base64
import io
from functools import partial

import kivy

kivy.require("2.1.0")


from kivy.app import App
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
from kivy.properties import StringProperty, ListProperty, ObjectProperty
from kivy.uix.screenmanager import Screen
from kivy.logger import Logger


from .utils import *

# from ..ml import detect_and_classify
detect_and_classify = lambda *args, **kwargs: None


Builder.load_file(str(pathlib.Path(__file__).parent / "menu.kv"))


class TrapSesionData(Widget):
    """
    One night / session of trap data.

    Will keep track of which directories have been processed, their cached results, etc.
    Could be backed by a SQLite database? Or just a folder structure under .cache
    """

    pass


class AnalyzeButton(Button):
    path = ObjectProperty()

    def on_release(self):
        self.analyze()

    def analyze(self):
        """
        Process an image directory.

        For now, just classify a random image.
        """
        annotations = detect_and_classify(self.path)
        # images = find_images(self.path)
        # img_path = random.choice(images)
        # results = predict_image(img_path)
        for widget in self.parent.children:
            # @TODO should we register nightly folders by ID somewhere?
            if isinstance(widget, Button):
                widget.disabled = False
        self.disabled = True
        # self.show_results(img_path, results)

    def show_results(self, img_path, results):
        content = GridLayout(rows=3, cols=1, spacing=20)
        img_container = RelativeLayout()
        img = Image(source=str(img_path), keep_ratio=True, allow_stretch=True)
        rect = Line(rounded_rectangle=(0, 0, 100, 100, 5))
        img.canvas.add(rect)
        img_container.add_widget(img)
        content.add_widget(img_container)

        result_text = Label(
            text=f"{results['highest_confidence_label']} \n{results['highest_confidence']}"
        )

        content.add_widget(result_text)

        close_button = Button(text="Close", size=(100, 20))
        # content.add_widget(close_button)

        popup = Popup(
            title="Classification results",
            content=content,
            auto_dismiss=True,
            size_hint=(None, None),
            size=(400, 400),
        )
        close_button.bind(on_press=popup.dismiss)
        popup.open()


class PlaybackButton(Button):
    path = ObjectProperty()
    screenmanager = ObjectProperty()

    def on_release(self):
        self.launch()

    def launch(self):
        """
        Review images and their annotations
        """

        self.screenmanager.current = "playback"
        self.screenmanager.get_screen("playback").source_dir = self.path


class SummaryButton(Button):
    path = ObjectProperty()
    screenmanager = ObjectProperty()

    def on_release(self):
        self.launch()

    def launch(self):
        """
        Open a the species summary screen
        """

        self.screenmanager.current = "summary"
        self.screenmanager.get_screen("summary").source_dir = self.path


class DataMenuScreen(Screen):
    root_dir = ObjectProperty(allownone=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Clock.schedule_once(self.scan_images, 1)

    def choose_root_directory(self, *args):
        self.root_dir = choose_directory(cache=False, starting_path=self.root_dir)
        self.scan_images()

    def scan_images(self, *args):
        if not self.root_dir:
            self.root_dir = choose_directory(cache=True)

        if self.root_dir:
            nightly_folders = find_timestamped_folders(self.root_dir)
            self.display_folders(nightly_folders)

    def display_folders(self, folders):
        print("Displaying folders")
        grid = self.ids.nightly_folders

        grid.clear_widgets()
        for date, path in folders.items():

            images = find_images(path)

            label = f"{date.strftime('%a, %b %-d')} \n{len(images)} images"
            bg_image = str(random.choice(images).absolute())

            annotations = find_annotations(path)
            btn_disabled = False if annotations else True

            analyze_btn = AnalyzeButton(
                text="Process", path=path, disabled=not btn_disabled
            )

            summary_btn = SummaryButton(
                text="Summary",
                path=path,
                screenmanager=self.manager,
                disabled=btn_disabled,
            )

            playback_btn = PlaybackButton(
                text="Playback",
                path=path,
                screenmanager=self.manager,
                disabled=btn_disabled,
            )

            row = GridLayout(rows=1, cols=5, spacing=10)
            row.add_widget(AsyncImage(source=bg_image))
            row.add_widget(Label(text=label))
            row.add_widget(analyze_btn)
            row.add_widget(summary_btn)
            row.add_widget(playback_btn)
            grid.add_widget(row)


class DataMenuApp(App):
    def build(self):
        return DataMenuScreen()


if __name__ == "__main__":
    DataMenuApp().run()