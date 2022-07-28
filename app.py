#! /usr/bin/env python3

import pathlib
import random
import tempfile
import dateutil.parser
import logging
import requests
import base64
import io

import kivy

kivy.require("2.1.0")


from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color, Canvas, Line
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
from kivy.properties import StringProperty, ListProperty, ObjectProperty

from utils import *
from ml import detect_and_classify


logger = logging.getLogger(__name__)

Builder.load_file("menu.kv")

# class Gallery(GridLayout):
#     images = ObjectProperty()
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         images = kwargs.get("images", [])
#         for image in images:
#             seld.add_widget(Image(source=str(image)))


class CanvasWidget(Widget):
    def __init__(self, **kwargs):

        super(CanvasWidget, self).__init__(**kwargs)
        image_path = kwargs.pop("image_path")

        # Arranging Canvas
        with self.canvas:

            Color(0.234, 0.456, 0.678, 0.8)  # set the colour

            # Setting the size and position of canvas
            self.rect = Rectangle(
                pos=self.center, size=(self.width / 2.0, self.height / 2.0)
            )
            self.bg = Image(
                source=str(image_path.absolute()), pos=self.pos, size=self.size
            )

            # Update the canvas as the screen size change
            self.bind(pos=self.update_rect, size=self.update_rect)

    # update function which makes the canvas adjustable.
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


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


class MainLayout(Widget):
    root_dir = ObjectProperty()

    def display_folders(self, folders):
        grid = self.ids.nightly_folders

        grid.children = []
        for date, path in folders.items():

            images = find_images(path)

            label = f"{date.strftime('%a, %b %-d')} \n{len(images)} images"
            bg_image = str(random.choice(images).absolute())

            analyze_btn = AnalyzeButton(text="Process", path=path)
            # analyze_btn.bind(on_release=self.analyze_callback)

            find_species_btn = Button(text="List species", disabled=True)
            play_video_btn = Button(text="Play video", disabled=True)

            row = GridLayout(rows=1, cols=5, spacing=10)
            row.add_widget(Image(source=bg_image))
            row.add_widget(Label(text=label))
            row.add_widget(analyze_btn)
            row.add_widget(find_species_btn)
            row.add_widget(play_video_btn)
            grid.add_widget(row)

    def draw_image(self, path):
        grid = self.ids.nightly_folders
        grid.children = []
        grid.add_widget(CanvasWidget(image_path=path))


class TrapDataAnalyzer(App):
    def build(self):
        self.title = "AMI Trap Data Companion"
        layout = MainLayout()
        self.root_dir = choose_root_directory(cache=False)
        if self.root_dir:
            nightly_folders = find_timestamped_folders(self.root_dir)
            layout.display_folders(nightly_folders)
        return layout


if __name__ == "__main__":
    TrapDataAnalyzer().run()
