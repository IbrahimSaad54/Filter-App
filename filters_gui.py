import cv2
import numpy as np
from tkinter import Tk, Button, filedialog
from PIL import Image, ImageTk
import tkinter as tk

def open_image():
    global img_original, img_display

    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.webp")])
    if path:
        img = cv2.imread(path)
        img_original = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_display = ImageTk.PhotoImage(img_pil.resize((400, 300)))
        panel.config(image=img_display)
        panel.image = img_display

def apply_cartoon():
    if img_original is not None:
        gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blurred, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_original, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        show_image(cartoon)

def apply_blur():
    if img_original is not None:
        blurred = cv2.GaussianBlur(img_original, (25, 25), 0)
        show_image(blurred)

def apply_edge():
    if img_original is not None:
        edges = cv2.Canny(img_original, 100, 300)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        show_image(edges_colored)

def apply_rotate():
    if img_original is not None:
        (h, w) = img_original.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
        rotated = cv2.warpAffine(img_original, matrix, (w, h))
        show_image(rotated)

def apply_translate():
    if img_original is not None:
        (h, w) = img_original.shape[:2]
        tx, ty = 100, 70
        matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        translated = cv2.warpAffine(img_original, matrix, (w, h))
        show_image(translated)

def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_display = ImageTk.PhotoImage(img_pil.resize((400, 300)))
    panel.config(image=img_display)
    panel.image = img_display

# ---------------- GUI Setup ----------------
img_original = None

root = Tk()
root.title("Image Filter App")

panel = tk.Label(root)
panel.pack()

btn_open = Button(root, text="📂 Load Image", command=open_image)
btn_open.pack(pady=5)

btn_cartoon = Button(root, text="🎨 Cartoon Effect", command=apply_cartoon)
btn_cartoon.pack(pady=2)

btn_blur = Button(root, text="🌫️ Heavy Blur", command=apply_blur)
btn_blur.pack(pady=2)

btn_edge = Button(root, text="⚡ Edge Detection", command=apply_edge)
btn_edge.pack(pady=2)

btn_rotate = Button(root, text="🔁 Rotate Image", command=apply_rotate)
btn_rotate.pack(pady=2)

btn_translate = Button(root, text="📦 Translate Image", command=apply_translate)
btn_translate.pack(pady=2)

root.mainloop()
