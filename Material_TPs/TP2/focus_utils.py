from functools import lru_cache

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def read_frames(path):
    """Devuelve (frames BGR apilados (N, H, W, 3), fps)."""
    cap = cv.VideoCapture(path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    return np.stack(frames), fps


def show(images, titles, cols=None, cmap="gray", size=3.5):
    """Grilla de imágenes. Si no es en blanco y negro se asumen BGR"""
    cols = cols or len(images)
    rows = int(np.ceil(len(images) / cols))
    fig, axs = plt.subplots(
        rows, cols, figsize=(size * cols, size * rows * 0.75), squeeze=False
    )
    for ax in axs.ravel():
        ax.axis("off")
    for ax, img, title in zip(axs.ravel(), images, titles):
        ax.imshow(
            cv.cvtColor(img, cv.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap=cmap
        )
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def spectrum(img):
    """Módulo del espectro centrado, en escala log para visualizarlo."""
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))


@lru_cache
def hann(shape):
    """Ventana de Hann 2D de tamaño `shape` (alto, ancho)."""
    return cv.createHanningWindow(shape[::-1], cv.CV_64F)


def gaussian_blur(img, sigma):
    return cv.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img
