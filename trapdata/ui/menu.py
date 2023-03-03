import pathlib
from typing import Optional

import kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.image import AsyncImage
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.recycleview import RecycleView
from kivy.lang import Builder
from kivy.properties import (
    StringProperty,
    ObjectProperty,
    BooleanProperty,
    ListProperty,
)
from kivy.uix.screenmanager import Screen
from plyer import filechooser

from trapdata import logger
from trapdata import db
from trapdata import TrapImage
from trapdata.db.models.queue import add_monitoring_session_to_queue
from trapdata.db.models.events import get_or_create_monitoring_sessions


kivy.require("2.1.0")


Builder.load_file(str(pathlib.Path(__file__).parent / "menu.kv"))


def choose_directory(use_saved: bool = True, starting_path: Optional[str] = None):
    """
    Prompt the user to select a directory where trap data has been saved.
    The subfolders of this directory should be timestamped directories
    with nightly trap images.

    The user's selection is saved and reused on the subsequent launch.
    """
    # @TODO Look for SDCARD / USB Devices first?

    app = App.get_running_app()
    image_base_path = app.config.get("paths", "image_base_path")
    selected_dir = None
    if image_base_path and use_saved:
        selected_dir = image_base_path

    else:
        selection = filechooser.choose_dir(
            title="Choose the root directory for your nightly trap data",
            path=starting_path,
        )

        if selection:
            selected_dir = selection[0]
            app.config.set("paths", "image_base_path", selected_dir)
            app.config.write()
            app.destroy_settings()

    return selected_dir


class AddToQueueButton(Button):
    monitoring_session = ObjectProperty()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_release(self):
        db_path = App.get_running_app().db_path
        add_monitoring_session_to_queue(db_path, self.monitoring_session)


class LaunchScreenButton(Button):
    monitoring_session = ObjectProperty()
    screen_name = StringProperty(allownone=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.screen_manager = App.get_running_app().screen_manager

    def on_release(self):
        self.launch()

    def launch(self):
        """
        Open the specified screen
        """

        if self.screen_manager and self.screen_name:
            self.screen_manager.current = self.screen_name
            self.screen_manager.get_screen(
                self.screen_name
            ).monitoring_session = self.monitoring_session


class MonitoringSessionRow(BoxLayout):
    monitoring_session = ObjectProperty()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = App.get_running_app()

    def on_monitoring_session(self, instance, value):
        ms = self.monitoring_session

        if ms:
            self.clear_widgets()

        with db.get_session(self.app.db_path) as sesh:
            first_image = (
                sesh.query(TrapImage)
                .filter_by(monitoring_session_id=ms.id)
                .order_by(TrapImage.filesize.desc())
                .first()
            )

        if not first_image:
            # If there is no first image
            logger.error(f"No images found for Monitoring Session: {ms}")
            return

        labels = [
            f"{ms.day.strftime('%a, %b %e')} \n{ms.duration_label} \n{ms.start_time.strftime('%H:%M')} - {ms.end_time.strftime('%H:%M')}",
            f"{ms.num_images or '0'} images\n",
        ]

        # btn_disabled = True
        btn_disabled = False

        playback_btn = LaunchScreenButton(
            text="Playback",
            monitoring_session=ms,
            screen_name="playback",
            disabled=btn_disabled,
        )

        add_to_queue_btn = AddToQueueButton(
            text="Add to Queue",
            monitoring_session=ms,
            disabled=btn_disabled,
        )

        summary_btn = LaunchScreenButton(
            text="Summary",
            monitoring_session=ms,
            screen_name="summary",
            disabled=btn_disabled,
        )

        buttons = [playback_btn, add_to_queue_btn, summary_btn]

        row = GridLayout(
            rows=1,
            cols=4 + len(labels),
            spacing=(0, 0),
            padding=(0, 0),
            # row_default_height=120,
            # row_force_default=True,
        )
        row.add_widget(
            AsyncImage(source=str(first_image.absolute_path), size_hint=(1, 1))
        )
        for label in labels:
            row.add_widget(Label(text=label, valign="top"))

        button_grid = GridLayout(rows=len(buttons), cols=1, padding=5, spacing=(0, 10))
        for button in buttons:
            button_grid.add_widget(button)

        row.add_widget(button_grid)

        self.add_widget(row)


class MonitoringSessionListView(RecycleView):
    monitoring_sessions = ListProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []

    def on_monitoring_sessions(self, instance, value):
        self.data = [{"monitoring_session": ms} for ms in self.monitoring_sessions]
        self.refresh_from_data()


class DataMenuScreen(Screen):
    image_base_path = ObjectProperty(allownone=True)
    sessions = ObjectProperty()
    status_popup = ObjectProperty()
    data_ready = BooleanProperty(defaultvalue=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = App.get_running_app()
        Clock.schedule_once(self.setup, 1)

    def setup(self, *args):
        last_directory = self.app.config.get("paths", "image_base_path")
        if last_directory:
            self.image_base_path = last_directory

    def choose_root_directory(self, *args):
        try:
            self.image_base_path = choose_directory(
                use_saved=False, starting_path=self.image_base_path
            )
        except Exception as e:
            logger.error(f"Failed to choose directory with a starting path: {e}")
            self.image_base_path = choose_directory(use_saved=False)

    def reload(self):
        """
        Reload the view by changing the root dir.
        """
        root_dir = self.image_base_path
        self.image_base_path = None
        self.image_base_path = root_dir

    def db_ready(self):
        # Try to open a database session.
        # # @TODO add GUI indicator asking to recreate DB if it fails to open?
        if not db.check_db(self.app.db_path, create=True, quiet=True):
            Popup(
                title="Error reading or creating database",
                content=Label(
                    text=(
                        f"Error reading or creating database: \n\n"
                        f"{self.app.db_path} \n\n"
                        f"Trying deleting the DB file and it will be recreated on next launch."
                    )
                ),
                size_hint=(None, None),
                size=("550dp", "200dp"),
                # on_dismiss=sys.exit,
            ).open()
            return False
        else:
            return True

    def on_image_base_path(self, instance, value):
        image_base_path = value
        logger.info("Base directory changed!")

        if image_base_path:
            self.app.image_base_path = image_base_path

            self.data_ready = False

            if image_base_path and self.db_ready():
                label_text = f"Looking for capture data in \n\n{image_base_path}"
                self.status_popup = Popup(
                    title="Status",
                    content=Label(text=label_text),
                    size_hint=(None, None),
                    size=("550dp", "200dp"),
                    auto_dismiss=False,
                    on_open=self.get_monitoring_sessions,
                )
                self.status_popup.open()

    def on_data_ready(self, *args):
        if self.data_ready:
            logger.info("Data is ready for other methods")
            # Buttons aren't available immediately
            # self.display_monitoring_sessions()
            # Clock.schedule_once(self.enable_buttons, 1)
        else:
            # self.disable_buttons()
            pass

    def enable_buttons(self, *args):
        logger.info("Enabling all buttons")
        for row in self.ids.monitoring_sessions.children:
            for child in row.children:
                if isinstance(child, Button):
                    child.disabled = False

    def disable_buttons(self, *args):
        logger.info("Disabling all buttons")
        for row in self.ids.monitoring_sessions.children:
            for child in row.children:
                if isinstance(child, Button):
                    child.disabled = True

    def get_monitoring_sessions(self, *args):
        sessions = get_or_create_monitoring_sessions(
            self.app.db_path, self.image_base_path
        )
        logger.info(
            f"Found {len(sessions)} monitoring sessions with base_path: {self.image_base_path}"
        )
        self.ids.monitoring_session_list.monitoring_sessions = sessions
        self.data_ready = True
        self.status_popup.dismiss()
        return sessions

    def open_settings(self):
        self.manager.current = "settings"
