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
from functools import partial
from threading import Thread

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

from .ml import detect_and_classify

# detect_and_classify = lambda *args, **kwargs: None


async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)


async def process_async_test(status):
    print(f"started at {time.strftime('%X')}")
    await say_after(1, "hello, one second")
    await say_after(2, "world, two seconds")
    print(f"completed at {time.strftime('%X')}")


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
    running = BooleanProperty(defaultvalue=False)
    progress = NumericProperty(defaultvalue=0)
    status_label = ObjectProperty()
    progress_label = ObjectProperty()
    popup = ObjectProperty()
    bgtask = ObjectProperty()

    def on_release(self):
        if not self.popup:
            self.make_popup()
        self.popup.open()

    def start(self, *args):
        # @TODO can use the status property of the bgtask thread instead
        if not self.running:
            self.running = True
            self.progress = 0
            self.bgtask = Thread(target=self.analyze)
            self.bgtask.start()

    def stop(self, *args):
        raise NotImplementedError
        if self.running:
            self.running = False
            if self.bgtask:
                self.bgtask.stop()

    def analyze(self):
        while self.progress < 20:
            time.sleep(0.5)
            self.progress += 1
        self.running = False

        # annotations = detect_and_classify(self.path)
        # images = find_images(self.path)
        # img_path = random.choice(images)
        # results = predict_image(img_path)
        # self.show_results(img_path, results)

    def complete(self):
        for widget in self.parent.children:
            # @TODO should we register nightly folders by ID somewhere?
            if isinstance(widget, Button):
                widget.disabled = False
        self.disabled = False

    def make_popup(self):
        if not self.popup:
            content = GridLayout(rows=5, cols=1, spacing=5)
            self.status_label = Label(text=f"Running: {self.running}")  # , id="status")
            self.progress_label = Label(
                text=f"Progress: {self.progress}"  # , id="progress"
            )
            content.add_widget(self.status_label)
            content.add_widget(self.progress_label)
            close_button = Button(text="Close progress", size=(100, 20))
            start_button = Button(text="Start", size=(100, 20))
            start_button.bind(on_press=self.start)
            stop_button = Button(text="Stop", size=(100, 20))
            stop_button.bind(on_press=self.stop)
            content.add_widget(start_button)
            content.add_widget(stop_button)
            content.add_widget(close_button)
            self.popup = Popup(
                title="Status",
                content=content,
                auto_dismiss=True,
                size_hint=(None, None),
                size=(400, 400),
            )
            close_button.bind(on_press=self.popup.dismiss)

    def on_progress(self, instance, value):
        self.progress_label.text = f"Progress: {value}"

    def on_running(self, instance, value):
        self.status_label.text = f"Running: {value}"


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
                text="Species List",
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
