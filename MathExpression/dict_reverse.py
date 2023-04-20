import tkinter as tk
from tkinter import ttk
import io
import numpy as np
from PIL import Image, ImageDraw, ImageTk

class PaintApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Paint App")
        self.canvas_width = 800
        self.canvas_height = 600

        self.draw_color = 'black'
        self.eraser_color = 'white'
        self.penwidth = 5

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.save_button = ttk.Button(self.master, text="Save", command=self.save_image)
        self.save_button.pack()

        self.erase_button = ttk.Button(self.master, text="Eraser", command=self.use_eraser)
        self.erase_button.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        self.canvas.create_oval(event.x - self.penwidth, event.y - self.penwidth, event.x + self.penwidth, event.y + self.penwidth, fill=self.draw_color, outline=self.draw_color)

    def reset(self, event):
        self.x = None
        self.y = None

    def use_eraser(self):
        self.draw_color = self.eraser_color

    def save_image(self):
        self.master.update()
        ps = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save('canvas.png')

if __name__ == '__main__':
    root = tk.Tk()
    PaintApp(root)
    root.mainloop()
