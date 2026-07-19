"""
test_export.py

Проверка формирования текстового и JSON-описания сцены.
"""

from pathlib import Path

import cv2

from scene.connected_components import extract_connected_components
from scene.description import describe_scene
from scene.export import export_json, export_txt


# ----------------------------------------------------
# Путь к тестовой маске
# ----------------------------------------------------

MASK_PATH = Path(
    "data/LoveDA_small/Val/Urban/masks_png/3516.png"
)

# ----------------------------------------------------

mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(
        f"Не удалось открыть маску:\n{MASK_PATH}"
    )

objects = extract_connected_components(
    mask,
    num_classes=8
)

description = describe_scene(
    objects,
    mask.shape
)

print()

print(description)

print()

export_txt(
    description,
    "scene.txt"
)

export_json(
    objects,
    mask.shape,
    "scene.json"
)

print("TXT сохранен.")

print("JSON сохранен.")