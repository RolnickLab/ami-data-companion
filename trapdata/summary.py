from functools import partial


from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image, AsyncImage
from kivy.lang import Builder
from kivy.clock import Clock

from .utils import *


Builder.load_file(str(pathlib.Path(__file__).parent / "summary.kv"))


class SpeciesSummaryScreen(Screen):
    source_dir = ObjectProperty()

    def on_source_dir(self, instance, value):
        species = self.load_species(self.source_dir)

    def add_row(self, layout, name, count, dt=None):
        text = f"{name} ({count})"
        label = Label(
            text=name,
            halign="left",
            valign="middle",
        )
        label.bind(size=label.setter("text_size"))
        # atlas_path = "home/michael/Projects/AMI/TRAPDATA/Vermont/2022_05_13/20220513211433-00-12.jpg"
        atlas_path = f"atlas://{self.source_dir}/trapdata/{slugify(name)}"
        # print("Atlas path", atlas_path)
        row = GridLayout(
            rows=1, cols=3, spacing=20, padding=20, height=220, size_hint_y=None
        )
        row.add_widget(Image(source=atlas_path, size_hint_y=None, height=150))
        row.add_widget(label)
        row.add_widget(Label(text=str(count), valign="top"))
        layout.add_widget(row)

    def load_species(self, path):
        # print(self.ids)
        layout = self.ids.species_list
        annotation_files = find_annotations(path)
        if not annotation_files:
            species = []
        else:
            species = summarize_species(annotation_files[0], best_only=True)
            fname = parse_annotations_to_kivy_atlas(annotation_files[0])

        layout.clear_widgets()
        layout.bind(minimum_height=layout.setter("height"))

        species_counts = [(name, details) for name, details in species.items()]
        species_counts.sort(key=lambda species: species[1]["count"], reverse=True)

        for i, (name, details) in enumerate(species_counts):
            Clock.schedule_once(
                partial(self.add_row, layout, name, details["count"]),
                i / len(species_counts),
            )

    def exit(self):
        self.manager.current = "menu"
