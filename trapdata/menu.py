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
    exit_event = ObjectProperty()
    progress_clock = ObjectProperty(allownone=True)

    def on_release(self):
        if not self.popup:
            self.make_popup()
        self.popup.open()

    def start(self, *args):
        # @TODO can use the status property of the bgtask thread instead
        if not self.running:
            self.running = True
            self.progress = 0
            self.exit_event = threading.Event()
            self.bgtask = threading.Thread(
                target=self.analyze,
                daemon=True,
                name=self.path.name,
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

    def analyze(self):
        # annotations = detect_and_classify(self.path)
        # while self.progress < 20:
        #     time.sleep(0.5)
        #     self.progress += 1
        #     if self.exit_event.is_set():
        #         break

        detect_and_classify(self.path)
        # @TODO this can't be in the separate bgtask thread because it modifies properties
        self.complete()
        return True
        # images = find_images(self.path)
        # img_path = random.choice(images)
        # results = predict_image(img_path)
        # self.show_results(img_path, results)

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
                title=f"Detect and classify moths in {self.path.name}",
                content=content,
                auto_dismiss=True,
                size_hint=(None, None),
                size=(400, 400),
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
    path = ObjectProperty()
    screenname = StringProperty(allownone=True)
    screenmanager = ObjectProperty(allownone=True)

    def on_release(self):
        self.launch()

    def launch(self):
        """
        Open a the the specified screen
        """

        if self.screenmanager and self.screenname:
            self.screenmanager.current = self.screenname
            self.screenmanager.get_screen(self.screenname).source_dir = self.path


class ReportButton(Button):
    path = ObjectProperty()

    def on_release(self):
        fname = export_report(self.path)
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
                text="Process",
                path=path,  # disabled=not btn_disabled
            )

            summary_btn = LaunchScreenButton(
                text="Species List",
                path=path,
                screenmanager=self.manager,
                screenname="summary",
                disabled=btn_disabled,
            )

            playback_btn = LaunchScreenButton(
                text="Playback",
                path=path,
                screenmanager=self.manager,
                screenname="playback",
                disabled=btn_disabled,
            )

            report_btn = ReportButton(
                text="Report",
                path=path,
                disabled=btn_disabled,
            )

            row = GridLayout(rows=1, cols=6, spacing=10)
            row.add_widget(AsyncImage(source=bg_image))
            row.add_widget(Label(text=label))
            row.add_widget(analyze_btn)
            row.add_widget(summary_btn)
            row.add_widget(playback_btn)
            row.add_widget(report_btn)
            grid.add_widget(row)


class DataMenuApp(App):
    def build(self):
        return DataMenuScreen()


if __name__ == "__main__":
    DataMenuApp().run()
