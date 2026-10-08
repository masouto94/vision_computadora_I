from pathlib import Path
from typing import NamedTuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class Box(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    score: float = 1.0  # |NCC| en [0, 1]
    sign: int = 1  # +1 misma polaridad que el template, -1 invertida

    @property
    def area(self):
        return self.width * self.height


def load_bgr(path):
    """Lee la imagen descartando el canal alfa (es 255 en todas)."""
    return cv.imread(str(path), cv.IMREAD_COLOR)


def load_images(folder):
    return {path.name: load_bgr(path) for path in sorted(Path(folder).iterdir())}


def crop_to_content(gray, threshold=200):
    """Recorta el margen claro alrededor del logo."""
    rows, cols = np.nonzero(gray < threshold)
    return gray[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def intersection_area(box, other):
    inter_width = max(0, min(box.x + box.width, other.x + other.width) - max(box.x, other.x))
    inter_height = max(0, min(box.y + box.height, other.y + other.height) - max(box.y, other.y))
    return inter_width * inter_height


def iou(box, other):
    """Intersección sobre unión de dos cajas."""
    inter = intersection_area(box, other)
    return inter / (box.area + other.area - inter)


def evaluate(detections, ground_truth, min_iou=0.5):
    """Asigna cada detección, de mayor a menor score, al logo libre de mayor IoU. Devuelve (TP, FP, FN)."""
    unmatched = list(ground_truth)
    hits = 0
    for detection in sorted(detections, key=lambda box: -box.score):
        closest = max(unmatched, key=lambda logo: iou(detection, logo), default=None)
        if closest is not None and iou(detection, closest) >= min_iou:
            unmatched.remove(closest)
            hits += 1
    return hits, len(detections) - hits, len(unmatched)


def draw_boxes(img, boxes, color=(0, 255, 0), labels=True):
    out = img.copy()
    thickness = max(2, round(max(img.shape) / 300))
    font_scale = max(0.4, max(img.shape) / 1200)
    for box in boxes:
        cv.rectangle(out, (box.x, box.y), (box.x + box.width, box.y + box.height), color, thickness)
        if labels:
            text = f"{box.score:.2f}"
            (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            above = box.y - text_height - 6 > 0
            baseline = box.y - 4 if above else box.y + box.height + text_height + 4
            cv.rectangle(out, (box.x, baseline - text_height - 3), (box.x + text_width + 2, baseline + 3), color, -1)
            cv.putText(out, text, (box.x + 1, baseline), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv.LINE_AA)
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
