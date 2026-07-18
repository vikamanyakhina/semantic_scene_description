"""
visualize_textures.py

Отображение исходного изображения и всех вычисленных текстурных признаков.
Используется для проверки корректности предварительно вычисленных текстур
и подготовки иллюстраций для отчета.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ------------------------------------------------------------
# ПУТИ
# ------------------------------------------------------------

DATASET_PATH = Path("/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA_small")

SPLIT = "Train"
AREA = "Urban"

IMAGE_NAME = ("3579    .png")          # поменять при необходимости


# ------------------------------------------------------------
# Загрузка данных
# ------------------------------------------------------------

image = np.array(
    Image.open(
        DATASET_PATH /
        SPLIT /
        AREA /
        "images_png" /
        IMAGE_NAME
    ).convert("RGB")
)

lbp = np.array(
    Image.open(
        DATASET_PATH /
        SPLIT /
        AREA /
        "textures" /
        "lbp" /
        IMAGE_NAME
    )
)

entropy = np.array(
    Image.open(
        DATASET_PATH /
        SPLIT /
        AREA /
        "textures" /
        "entropy" /
        IMAGE_NAME
    )
)

variance = np.array(
    Image.open(
        DATASET_PATH /
        SPLIT /
        AREA /
        "textures" /
        "variance" /
        IMAGE_NAME
    )
)

gabor = np.array(
    Image.open(
        DATASET_PATH /
        SPLIT /
        AREA /
        "textures" /
        "gabor" /
        IMAGE_NAME
    )
)


# ------------------------------------------------------------
# Визуализация
# ------------------------------------------------------------

plt.figure(figsize=(18, 8))

titles = [
    "RGB",
    "LBP",
    "Entropy",
    "Variance",
    "Gabor"
]

images = [
    image,
    lbp,
    entropy,
    variance,
    gabor
]

for i, (title, img) in enumerate(zip(titles, images), start=1):

    plt.subplot(2, 3, i)

    if i == 1:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")

    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()