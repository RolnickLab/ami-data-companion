import pathlib
from functools import partial

from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from kivy.clock import Clock

from trapdata import logger
from trapdata.models.queue import all_queues


Builder.load_file(str(pathlib.Path(__file__).parent / "queue.kv"))


class QueueStatusTable(BoxLayout):
    refresh_clock = ObjectProperty(allownone=True)

    def start_auto_refresh(self):
        refresh_interval_seconds = 1

        if self.refresh_clock:
            Clock.unschedule(self.refresh_clock)

        logger.debug(
            f"Scheduling auto-refresh of queue status every {refresh_interval_seconds} seconds"
        )
        self.refresh_clock = Clock.schedule_interval(
            self.refresh, refresh_interval_seconds
        )

    def stop_auto_refresh(self):
        logger.debug("Stopping auto-refresh of queue status list")
        if self.refresh_clock:
            Clock.unschedule(self.refresh_clock)

    def refresh(self, *args):
        # logger.debug("Refreshing queue status view")
        self.clear_widgets()

        data = self.get_queue_data()
        num_rows = len(data) + 3
        num_cols = max([len(row) for row in data])
        table = GridLayout(
            rows=num_rows,
            cols=num_cols,
            padding=0,
            spacing=0,
            row_default_height=80,
            row_force_default=True,
        )

        header = data.pop(0)

        for field_value in header:
            text = f"[b]{field_value.upper()}[/b]"
            label = Label(text=text, markup=True)
            table.add_widget(label)

        for row in data:
            actions = row.pop()
            for field_value in row:
                text = str(field_value)
                label = Label(
                    text=text,
                    halign="left",
                    markup=True,
                )
                table.add_widget(label)
            for btn in actions:
                table.add_widget(btn)

        self.add_widget(table)

    def get_queue_data(self):
        """
        Return the details of each queue the format that the "viewclass" widget
        expects. In this case the viewclass is a `InfoRow` object.
        """

        rows = [
            [
                "Data Type",
                "Unprocessed",
                "Queued",
                "Complete",
                "Status",
                "",
                "",
            ]
        ]
        app = App.get_running_app()

        for name, queue in all_queues(app.base_path).items():
            clear_button = Button(text="Clear")
            clear_button.bind(on_release=partial(queue.clear_queue))
            add_button = Button(text="Add")  # Add remaining unprocessed
            add_button.bind(on_release=partial(queue.add_unprocessed))
            rows.append(
                [
                    name,
                    queue.unprocessed_count(),
                    queue.queue_count(),
                    queue.done_count(),
                    "Running" if app.queue and app.queue.running else "Stopped",
                    [add_button, clear_button],
                ]
            )
        return rows


class QueueScreen(Screen):
    def exit(self):
        self.manager.current = "menu"

    def refresh(self):
        self.ids.queue_status.refresh()

    def on_enter(self, *args):
        self.ids.queue_status.refresh()
        self.ids.queue_status.start_auto_refresh()

    def on_leave(self, *args):
        self.ids.queue_status.stop_auto_refresh()
