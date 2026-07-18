"""
textures/lbp.py

Вычисление карты локальных бинарных шаблонов (LBP).
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def compute_lbp(image: np.ndarray,
                radius: int = 1,
                n_points: int = 8) -> np.ndarray:
    """
    Вычисляет карту LBP.

    Parameters
    ----------
    image : np.ndarray
        RGB изображение.

    radius : int
        Радиус окрестности.

    n_points : int
        Количество соседних пикселей.

    Returns
    -------
    np.ndarray
        Нормализованная карта LBP.
    """

    # Переводим изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Вычисляем LBP
    lbp = local_binary_pattern(
        gray,
        P=n_points,
        R=radius,
        method="uniform"
    )

    # Нормализуем значения в диапазон 0..255
    lbp = cv2.normalize(
        lbp,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return lbp.astype(np.uint8)