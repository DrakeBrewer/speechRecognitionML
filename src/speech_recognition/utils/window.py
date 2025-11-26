import tkinter as tk
import colorsys
from typing import Callable, Tuple

class Window: 
    def __init__(
            self, 
            title: str,
            dimensions: str,
            bg_color: str,
            update_func: Callable[[], Tuple[str, float]]
    ):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(background=bg_color)
        self.root.geometry(dimensions)
        self.update_func = update_func

        self.__draw_canvas(bg_color)

    def __draw_canvas(self, bg_color: str):
        self.canvas = tk.Canvas(self.root, bg=bg_color, highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        self.text_name = self.canvas.create_text(
            0, 0,
            text="0",
            anchor="center",
            fill="#eaeaea",
            font=("Roboto", 28)
        )

        self.text_conf = self.canvas.create_text(
            0, 0,
            text="0",
            anchor="center",
            fill="#eaeaea",
            font=("Roboto", 28)
        )

    def layout(self):
        xCenter = self.root.winfo_width() // 2
        yCenter = self.root.winfo_height() // 2

        self.canvas.coords(self.text_name, xCenter, yCenter)
        self.canvas.coords(self.text_conf, xCenter, yCenter + 50)

    def update(self) -> None:
        self.layout()

        label, confidence = self.update_func()

        self.canvas.itemconfigure(self.text_name, text=label)
        self.canvas.itemconfigure(self.text_conf, text=f"{confidence:.2f}", fill=color_conf(confidence))
        self.root.after(100, self.update)

    def run(self):
        self.update()
        self.root.mainloop()



def color_conf(value: float) -> str:
    hue = value * 120
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.5, 1)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

