"""
textures/variance.py

Вычисление карты локальной дисперсии.
"""

import cv2
import numpy as np


def compute_variance(image: np.ndarray,
                     kernel_size: int = 9) -> np.ndarray:
    """
    Вычисляет карту локальной дисперсии.

    Parameters
    ----------
    image : np.ndarray
        RGB изображение.

    kernel_size : int
        Размер окна.

    Returns
    -------
    np.ndarray
        Карта локальной дисперсии.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = gray.astype(np.float32)

    # Локальное среднее
    mean = cv2.blur(gray, (kernel_size, kernel_size))

    # Локальное среднее квадрата
    mean_sq = cv2.blur(gray ** 2, (kernel_size, kernel_size))

    # Var(X)=E(X²)-E(X)²
    variance = mean_sq - mean ** 2

    variance = cv2.normalize(
        variance,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return variance.astype(np.uint8)