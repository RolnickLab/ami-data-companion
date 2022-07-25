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
from kivy.graphics import Rectangle, Color
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from kivy.properties import StringProperty, ListProperty, ObjectProperty

from plyer import filechooser
import PIL


logger = logging.getLogger(__name__)

Builder.load_file("menu.kv")

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg")

TEMPORARY_BASE_PATH = "/media/michael/LaCie/AMI/"


# class Gallery(GridLayout):
#     images = ObjectProperty()
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         images = kwargs.get("images", [])
#         for image in images:
#             seld.add_widget(Image(source=str(image)))


def cache_dir():
    # If fails, use temp dir?
    # d = tempfile.TemporaryDirectory(delete=False)
    d = pathlib.Path(".cache")
    d.mkdir(exist_ok=True)
    return d


def save_setting(key, val):
    """
    >>> save_setting("last_test", "now")
    'now'
    >>> read_setting("last_test")
    'now'
    """
    f = cache_dir() / key
    logger.debug(f"Writing to cache: {f}")
    f.write_text(val)
    return val


def read_setting(key):
    f = cache_dir() / key
    logger.debug(f"Checking cache: {f}")
    if f.exists():
        return f.read_text()
    else:
        return None


def delete_setting(key):
    f = cache_dir() / key
    logger.debug(f"Deleting cache: {f}")
    if f.exists():
        return f.unlink()
    else:
        return None


def choose_root_directory():
    """
    Prompt the user to select a directory where trap data has been saved.
    The subfolders of this directory should be timestamped directories
    with nightly trap images.

    The user's selection is saved and reused on the subsequent launch.
    """
    # @TODO Look for SDCARD / USB Devices first?

    setting_key = "last_root_dir"

    root_dir = read_setting(setting_key)

    if root_dir:
        root_dir = pathlib.Path(root_dir)

        if root_dir.is_dir():
            return root_dir
        else:
            delete_setting(setting_key)

    selection = filechooser.choose_dir(
        title="Choose the root directory for your nightly trap data",
        path=TEMPORARY_BASE_PATH,
    )

    if selection:
        root_dir = selection[0]
    else:
        return None

    save_setting(setting_key, root_dir)

    return root_dir


def find_timestamped_folders(path):
    """
    Find all directories in a given path that have
    dates / timestamps in the name.

    This should be the nightly folders from the trap data.

    >>> pathlib.Path("./tmp/2022_05_14").mkdir(exist_ok=True, parents=True)
    >>> pathlib.Path("./tmp/nope").mkdir(exist_ok=True, parents=True)
    >>> find_timestamped_folders("./tmp")
    [PosixPath('tmp/2022_05_14')]
    """
    nights = {}

    def _preprocess(name):
        return name.replace("_", "-")

    for d in pathlib.Path(path).iterdir():
        # @TODO use yield?
        try:
            date = dateutil.parser.parse(_preprocess(d.name))
        except Exception:
            # except dateutil.parser.ParserError:
            pass
        else:
            nights[date] = d

    # @TODO should be sorted by date
    return nights


def find_images(path):
    images = [
        f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return images


def predict_image(path):
    img = PIL.Image.open(path)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    resp = requests.post("http://localhost:5000/predict", data={"b64": img_str})
    resp.raise_for_status()
    results = resp.json()
    return results


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
        images = find_images(self.path)
        img_path = random.choice(images)
        results = predict_image(img_path)
        for widget in self.parent.children:
            # @TODO should we register nightly folders by ID somewhere?
            if isinstance(widget, Button):
                widget.disabled = False
        self.show_results(img_path, results)

    def show_results(self, img_path, results):
        content = GridLayout(rows=3, cols=1, spacing=20)
        content.add_widget(Image(source=str(img_path)))

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
        self.root_dir = choose_root_directory()
        if self.root_dir:
            nightly_folders = find_timestamped_folders(self.root_dir)
            layout.display_folders(nightly_folders)
        return layout


if __name__ == "__main__":
    TrapDataAnalyzer().run()