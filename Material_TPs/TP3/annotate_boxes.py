"""Anotación manual del ground truth.

Uso, desde Material_TPs/TP3:
    python annotate_boxes.py                 # todas las imágenes
    python annotate_boxes.py coca_multi.png  # sólo algunas; el resto se conserva

En cada ventana: arrastrar un rectángulo por logo y confirmar cada uno con ENTER o ESPACIO.
Cuando no quedan logos, ESC pasa a la siguiente imagen. Para descartar la última caja, tecla c.
Las cajas se guardan en ground_truth.json como [x, y, ancho, alto] en coordenadas de la imagen original.
"""
import json
import sys
from pathlib import Path

import cv2 as cv

IMAGES = Path("images")
OUTPUT = Path("ground_truth.json")
MAX_SIDE = 900  # las imágenes grandes se muestran reducidas para que entren en pantalla


def annotate(path):
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    zoom = min(1.0, MAX_SIDE / max(image.shape[:2]))
    view = cv.resize(image, None, fx=zoom, fy=zoom, interpolation=cv.INTER_AREA) if zoom < 1 else image
    title = f"{path.name} - ENTER confirma caja, ESC termina"
    selections = cv.selectROIs(title, view, showCrosshair=False)
    cv.destroyWindow(title)
    boxes = []
    for selection in selections:
        box = [round(value / zoom) for value in selection]
        if box not in boxes:  # confirmar dos veces la misma caja no la duplica
            boxes.append(box)
    return boxes


def save(annotations):
    entries = [f'  "{name}": [\n' + ",\n".join(f"    {box}" for box in boxes) + "\n  ]"
               for name, boxes in annotations.items()]
    OUTPUT.write_text("{\n" + ",\n".join(entries) + "\n}\n", encoding="utf8")


def main():
    annotations = json.loads(OUTPUT.read_text(encoding="utf8")) if OUTPUT.exists() else {}
    names = sys.argv[1:] or sorted(path.name for path in IMAGES.iterdir())
    for name in names:
        annotations[name] = annotate(IMAGES / name)
        print(f"{name}: {len(annotations[name])} logos -> {annotations[name]}")
        save(annotations)  # se guarda después de cada imagen
    print(f"guardado en {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
