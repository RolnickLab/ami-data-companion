import json
import pathlib
import random


import kivy

kivy.require("2.1.0")


from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import Rectangle, Color, Canvas, Line, Ellipse
from kivy.core.window import Window

Builder.load_file("image_bbox_test.kv")


def load_sample():
    source_dir = pathlib.Path(
        # "/media/michael/LaCie/AMI/TrapData_2022/Quebec/2022_05_17"
        "/media/michael/LaCie/AMI/TrapData_2022/Vermont/test"
    )
    annotations = json.load(
        open(
            "/media/michael/LaCie/AMI/TrapData_2022/Vermont/localize_classify_annotation-test.json"
        )
    )
    sample = random.choice(list(annotations.keys()))
    img_path = source_dir / sample
    bboxes = annotations[sample][0]
    labels = annotations[sample][1]
    return img_path, bboxes


class CanvasWidget(Widget):
    # image_path = StringProperty()
    # bboxes = ListProperty()

    def __init__(self, **kwargs):

        self.image_path = kwargs.pop("image_path")
        self.bboxes = kwargs.pop("bboxes")

        super(CanvasWidget, self).__init__(**kwargs)

        # Arranging Canvas
        with self.canvas:

            # Update the canvas as the screen size change
            self.bind(pos=self.update_rect, size=self.update_rect)

        self.draw()

    # update function which makes the canvas adjustable.
    def update_rect(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size
        self.draw()

    def draw(self, *args):
        self.canvas.clear()
        with self.canvas:
            img = Image(
                source=str(self.image_path.absolute()),
                pos=(0, 0),
                size=self.size,
                pos_hint={"bottom": 0},
                keep_ratio=False,
                allow_stretch=True,
            )
            print("Image sizes:")
            print(img.width, img.size, img.norm_image_size, img.texture_size)
            print()
            win_width, win_height = img.norm_image_size
            img_width, img_height = img.texture_size
            x_scale = win_width / img_width
            y_scale = win_height / img_height
            print("x scale:", x_scale, "y_scale:", y_scale)
            self.bg = img
            self.bbox_widgets = []
            for i, box in enumerate(self.bboxes):
                Color(1, 0, 0, 1)
                self.bbox_widgets.append(
                    Line(rectangle=(0, 0, win_width, win_height), width=2)
                )
                print("bbox#", i)

                color = [random.random() for _ in range(3)]
                color.append(0.8)  # alpha
                Color(*color)
                x1, y1, x2, y2 = box

                w = x2 - x1
                h = y2 - y1
                print("original dims:", w, h)

                # Reference from bottom left instead of top left
                y1 = img_height - y1
                y2 = img_height - y2

                # Scale bbox to match on-screen scale of image
                x1 *= x_scale
                y1 *= y_scale
                x2 *= x_scale
                y2 *= y_scale

                w2 = w * x_scale
                h2 = h * y_scale
                print("new dims:", w2, h2)

                w2 = x2 - x1
                h2 = y1 - y2
                print("new dims by coord:", w2, h2)

                rect = (x1, y2, x2, y1)
                print(box, rect)
                self.bbox_widgets.append(
                    Line(points=[x1, y1, x1, y2, x2, y2, x2, y1, x1, y1], width=2)
                )
                self.bbox_widgets.append(Label(text=str(i), pos=(x1, y2), color=color))


class MainLayout(BoxLayout):
    pass


class BBox(BoxLayout):
    pass


class ImagePreview(Image):
    pass


class PreviewWindow(RelativeLayout):
    def random_sample(self):
        # show_image_with_annotations
        img_path, bboxes = load_sample()
        self.source = ""  # str(img_path.absolute())
        self.children = []
        cvs = CanvasWidget(
            image_path=img_path, bboxes=bboxes, size=self.size, pos_hint={"bottom": 0}
        )
        self.add_widget(cvs)


class Controls(BoxLayout):
    pass


class ImageOverlayApp(App):
    def build(self):
        self.title = "Image bbox overlay test"
        layout = MainLayout()
        Window.clearcolor = (0, 1, 0, 1)

        label = Label(size_hint=(1, 0.05))
        Window.bind(mouse_pos=lambda w, p: setattr(label, "text", str(p)))
        layout.add_widget(label)

        return layout


if __name__ == "__main__":
    ImageOverlayApp().run()