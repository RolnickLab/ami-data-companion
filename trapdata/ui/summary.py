import pathlib

from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.clock import Clock

from trapdata import logger
from trapdata.db import queries


Builder.load_file(str(pathlib.Path(__file__).parent / "summary.kv"))


class SpeciesRow(BoxLayout):
    species = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_species(self, instance, value):
        if self.species:
            self.make_row(self.species)

    def make_row(self, species):
        self.clear_widgets()
        label = Label(
            text=species["name"],
            halign="left",
            valign="middle",
        )
        label.bind(size=label.setter("text_size"))

        # self.add_widget(
        #     Image(
        #         source=species["atlas_path"],
        #         size_hint_y=None,
        #         height=species["image_height"],
        #     )
        # )
        self.add_widget(label)
        self.add_widget(Label(text=str(species["count"]), valign="top"))


class SpeciesListLayout(RecycleView):
    monitoring_session = ObjectProperty()
    refresh_clock = ObjectProperty(allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []

    def on_monitoring_session(self, instance, value):
        self.refresh()
        self.start_auto_refresh()

    def start_auto_refresh(self):
        refresh_interval_seconds = 1

        if self.refresh_clock:
            Clock.unschedule(self.refresh_clock)

        logger.debug(
            f"Scheduling auto-refresh of species list every {refresh_interval_seconds} seconds"
        )
        self.refresh_clock = Clock.schedule_interval(
            self.refresh, refresh_interval_seconds
        )

    def stop_auto_refresh(self):
        logger.debug("Stopping auto-refresh of species list")
        if self.refresh_clock:
            Clock.unschedule(self.refresh_clock)

    def refresh(self, *args):
        # logger.debug("Refreshing species list")
        self.data = self.load_species(self.monitoring_session)
        self.refresh_from_data()

    def load_species(self, ms):
        """
        Return a list of species in the format that the "viewclass" widget
        expects. In this case the viewclass is a `SpeciesRow` object.
        """
        app = App.get_running_app()
        species_counts = queries.count_species(app.db_path, ms)

        row_height = 100  # @TODO make dynamic? Or fit images to specific size

        widget_attrs = [
            {
                "species": {
                    "name": label.title() if label else "Unclassified",
                    "count": count,
                    "image_height": row_height,
                },
                "height": row_height,
            }
            for label, count in species_counts
        ]

        return widget_attrs


class SpeciesSummaryScreen(Screen):
    monitoring_session = ObjectProperty()

    def on_monitoring_session(self, instance, value):
        """
        Update the monitoring session of the child SpeciesListLayout
        when the monitoring session of this screen changes.
        """
        self.ids.species_list.monitoring_session = self.monitoring_session

    def exit(self):
        self.manager.current = "menu"

    def reload(self):
        self.ids.species_list.refresh()

    def on_enter(self, *args):
        self.ids.species_list.start_auto_refresh()

    def on_leave(self, *args):
        self.ids.species_list.stop_auto_refresh()
