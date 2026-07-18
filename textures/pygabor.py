"""
textures/gabor.py
"""

import cv2
import numpy as np


def compute_gabor(image: np.ndarray):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    responses = []

    thetas = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    for theta in thetas:

        kernel = cv2.getGaborKernel(
            (21, 21),
            5,
            theta,
            10,
            0.5,
            0,
            cv2.CV_32F
        )

        response = cv2.filter2D(
            gray,
            cv2.CV_32F,
            kernel
        )

        responses.append(response)

    gabor = np.max(responses, axis=0)

    gabor = cv2.normalize(
        gabor,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return gabor.astype(np.uint8)