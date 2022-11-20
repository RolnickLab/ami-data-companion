import pathlib
import time

from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image, AsyncImage
from kivy.lang import Builder
from kivy.clock import Clock

from trapdata import logger
from trapdata.db.models.detections import (
    get_unique_species,
    get_objects_for_species,
)


Builder.load_file(str(pathlib.Path(__file__).parent / "species_summary.kv"))


class SpeciesGrid(StackLayout):
    species = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_species(self, instance, value):
        if self.species:
            app = App.get_running_app()
            detections = get_objects_for_species(
                app.db, self.species["name"], self.species["monitoring_session"]
            )
            logger.info(f"Showing all detections for species: {self.species['name']}")
            self.make_row(self.species, detections)

    def make_row(self, species, detections):
        self.clear_widgets()
        label = Label(
            text=species["name"],
            halign="left",
            valign="top",
            size_hint_x=0.2,
        )
        label.bind(size=label.setter("text_size"))

        # self.add_widget(label)

        stack = StackLayout(
            orientation="lr-tb",
            spacing=(0, 0),
            padding=(0, 0),
            size_hint_x=0.8,
            size_hint_y=None,
        )

        for row in detections[:5]:
            obj = row[0]
            self.add_widget(
                Image(
                    source=obj.path,
                    width=100,
                    height=100,
                )
            )
        # self.add_widget(stack)


class SpeciesGridLayout(RecycleView):
    monitoring_session = ObjectProperty()
    refresh_clock = ObjectProperty(allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []

    def on_monitoring_session(self, instance, value):
        self.refresh()
        # self.start_auto_refresh()

    def start_auto_refresh(self):
        refresh_interval_seconds = 1

        if self.refresh_clock:
            Clock.unschedule(self.refresh_clock)

        logger.debug(
            f"Scheduling auto-refresh of species grid every {refresh_interval_seconds} seconds"
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
        species = get_unique_species(app.db, ms)
        logger.info(
            f"Found {len(species)} unique species in monitoring session {ms.id}"
        )

        widget_attrs = [
            {
                "species": {
                    "name": label or "Unclassified",
                    "count": count,
                    "monitoring_session": ms,
                },
                "height": 300,
                "width": 1000,
                "size_hint_y": None,
                "size_hint_x": 1,
            }
            for label, count in species
        ]

        return widget_attrs


class SpeciesSummaryGridScreen(Screen):
    monitoring_session = ObjectProperty()

    def on_monitoring_session(self, instance, value):
        """
        Update the monitoring session of the child SpeciesListLayout
        when the monitoring session of this screen changes.
        """
        self.ids.species_grid.monitoring_session = self.monitoring_session

    def exit(self):
        self.manager.current = "menu"

    def refresh(self):
        self.ids.species_grid.refresh()

    # def on_enter(self, *args):
    #     self.ids.species_grid.start_auto_refresh()

    def on_leave(self, *args):
        self.ids.species_grid.stop_auto_refresh()
