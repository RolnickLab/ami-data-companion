from functools import partial


from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image, AsyncImage
from kivy.lang import Builder
from kivy.clock import Clock

from .utils import *


Builder.load_file(str(pathlib.Path(__file__).parent / "summary.kv"))


class SpeciesRow(BoxLayout):
    species = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_species(self, instance, value):
        self.make_row(value)

    def make_row(self, species):
        self.clear_widgets()
        label = Label(
            text=species["name"],
            halign="left",
            valign="middle",
        )
        label.bind(size=label.setter("text_size"))

        self.add_widget(
            Image(
                source=species["atlas_path"],
                size_hint_y=None,
                height=species["image_height"],
            )
        )
        self.add_widget(label)
        self.add_widget(Label(text=str(species["count"]), valign="top"))


class SpeciesListLayout(RecycleView):
    source_dir = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []

    def on_source_dir(self, instance, value):
        print("SPECIES LAYOUT SOURCE_DIR CHANGED!", value)
        self.data = self.load_species(value)
        self.refresh_from_data()

    def load_species(self, path):
        """
        Return a list of species in the format that the "viewclass" widget
        expects. In this case the viewclass is a `SpeciesRow` object.
        """
        if not path:
            return []

        annotation_files = find_annotations(path)
        if not annotation_files:
            species = []
        else:
            species = summarize_species(annotation_files[0], best_only=True)
            fname = parse_annotations_to_kivy_atlas(annotation_files[0])

        species_counts = [(name, details) for name, details in species.items()]
        species_counts.sort(key=lambda species: species[1]["count"], reverse=True)

        row_height = 200  # @TODO make dynamic? Or fit images to specific size
        widget_attrs = [
            {
                "species": {
                    "atlas_path": f"atlas://{self.source_dir}/trapdata/{slugify(name)}",
                    "name": name,
                    "count": details["count"],
                    "image_height": row_height,
                },
                "height": row_height,
            }
            for name, details in species_counts
        ]

        return widget_attrs


class SpeciesSummaryScreen(Screen):
    source_dir = ObjectProperty()

    def on_source_dir(self, instance, value):
        """
        Update the source dir of the child SpeciesListLayout
        when the source_dir of this screen changes.
        """
        self.ids.species_list.source_dir = self.source_dir

    def exit(self):
        self.manager.current = "menu"
