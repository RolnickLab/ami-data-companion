import pathlib
import random
import tempfile
import dateutil.parser
import logging


import kivy

kivy.require("2.1.0")


from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder

from plyer import filechooser


logger = logging.getLogger(__name__)

Builder.load_file("menu.kv")

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg")


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
        except dateutil.parser.ParserError:
            pass
        else:
            nights[date] = d

    # @TODO should be sorted by date
    return nights


class MainLayout(Widget):
    def choose_root_directory(self):
        # @TODO "$HOME" doesn't work, maybe choose SDCARD / USB Device first?
        # /home/michael/Projects/AMI/data/TrapData_2022

        root_dir = read_setting("last_root_dir") or save_setting(
            "last_root_dir",
            filechooser.choose_dir(
                title="Choose the root directory for your nightly trap data",
                path="$HOME",
            )[0],
        )

        nightly_folders = find_timestamped_folders(root_dir)
        self.display_folders(nightly_folders)

    def display_folders(self, folders):

        grid = self.ids.nightly_folders

        grid.children = []
        for date, path in folders.items():
            images = [
                f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            bg_image = str(random.choice(images).absolute())
            label = f"{date.strftime('%c')} \n({len(images)} images)"
            button = Button(
                text=label,
                # color=(0, 0, 1, 1),
                # color=(0, 0, 0, 1),
                # bold=True,
                font_size="24sp",
                # outline_color=(1, 1, 1),
                # outline_width=1,
                # background_normal=bg_image,
                # border = (30, 30, 30, 30),
            )
            callback = lambda btn: print(path, btn)
            button.bind(on_release=callback)
            grid.add_widget(button)


class TrapDataAnalyzer(App):
    def build(self):
        layout = MainLayout()
        layout.choose_root_directory()
        return layout


if __name__ == "__main__":
    TrapDataAnalyzer().run()
