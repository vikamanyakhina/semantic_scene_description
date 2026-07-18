"""
textures/entropy.py

Вычисление карты локальной энтропии изображения.
"""

import cv2
import numpy as np

from skimage.filters.rank import entropy
from skimage.morphology import disk


def compute_entropy(image: np.ndarray,
                    radius: int = 5) -> np.ndarray:
    """
    Вычисляет карту локальной энтропии.

    Parameters
    ----------
    image : np.ndarray
        RGB изображение.

    radius : int
        Радиус локального окна.

    Returns
    -------
    np.ndarray
        Карта энтропии.
    """

    # Переводим изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Вычисляем локальную энтропию
    entropy_map = entropy(gray, disk(radius))

    # Нормализуем значения
    entropy_map = cv2.normalize(
        entropy_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return entropy_map.astype(np.uint8)