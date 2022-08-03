from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.lang import Builder

from .utils import *


Builder.load_file(str(pathlib.Path(__file__).parent / "summary.kv"))


class SpeciesSummaryScreen(Screen):
    source_dir = ObjectProperty()

    def on_source_dir(self, instance, value):
        self.load_species(self.source_dir)

    def load_species(self, path):
        print(self.ids)
        layout = self.ids.species_list
        annotation_files = find_annotations(path)
        if not annotation_files:
            species = []
        else:
            species = summarize_species(annotation_files[0])

        layout.clear_widgets()
        for name, details in species.items():
            text = f"{name} ({len(details)})"
            line = Label(text=text, height=20)
            layout.add_widget(line)

    def exit(self):
        self.manager.current = "menu"
