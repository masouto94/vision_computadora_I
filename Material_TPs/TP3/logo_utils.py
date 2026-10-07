from pathlib import Path
from typing import NamedTuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class Box(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0  # |NCC| en [0, 1]
    sign: int = 1  # +1 misma polaridad que el template, -1 invertida


def load_bgr(path):
    """Lee la imagen descartando el canal alfa (es 255 en todas)."""
    return cv.imread(str(path), cv.IMREAD_COLOR)


def load_images(folder):
    return {p.name: load_bgr(p) for p in sorted(Path(folder).iterdir())}


def crop_to_content(gray, thresh=200):
    """Recorta el margen claro alrededor del logo."""
    ys, xs = np.nonzero(gray < thresh)
    return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def iou(a, b):
    iw = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    ih = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = iw * ih
    return inter / (a.w * a.h + b.w * b.h - inter)


def evaluate(dets, gts, thr=0.5):
    """Asigna cada detección (de mayor a menor score) al GT libre de mayor IoU. Devuelve (TP, FP, FN)."""
    free = list(gts)
    tp = 0
    for d in sorted(dets, key=lambda b: -b.score):
        best = max(free, key=lambda g: iou(d, g), default=None)
        if best is not None and iou(d, best) >= thr:
            free.remove(best)
            tp += 1
    return tp, len(dets) - tp, len(free)


def draw_boxes(img, boxes, color=(0, 255, 0), labels=True):
    out = img.copy()
    t = max(2, round(max(img.shape) / 300))
    fs = max(0.4, max(img.shape) / 1200)
    for b in boxes:
        cv.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), color, t)
        if labels:
            txt = f"{b.score:.2f}"
            (tw, th), _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, fs, 1)
            y = b.y - 4 if b.y - th - 6 > 0 else b.y + b.h + th + 4
            cv.rectangle(out, (b.x, y - th - 3), (b.x + tw + 2, y + 3), color, -1)
            cv.putText(out, txt, (b.x + 1, y), cv.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv.LINE_AA)
    return out


def show(images, titles, cols=None, size=4, cmap="gray"):
    """Grilla de imágenes. Si no es en blanco y negro se asumen BGR."""
    cols = cols or len(images)
    rows = int(np.ceil(len(images) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(size * cols, size * rows * 0.8), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, img, title in zip(axs.ravel(), images, titles):
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap=cmap)
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()
