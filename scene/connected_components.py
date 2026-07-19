"""
scene/connected_components.py

Выделение связных областей на маске семантической сегментации.
Для каждого класса находятся отдельные объекты.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class SceneObject:
    """
    Описание одного объекта сцены.
    """

    class_id: int

    object_id: int

    mask: np.ndarray

    area_pixels: int

    bbox: tuple

    centroid: tuple


def extract_connected_components(mask: np.ndarray,
                                 num_classes: int,
                                 min_area: int = 30) -> List[SceneObject]:
    """
    Выделяет связные области для каждого класса.

    Parameters
    ----------
    mask : np.ndarray

        Маска сегментации.

    num_classes : int

        Количество классов.

    min_area : int

        Минимальная площадь объекта.

    Returns
    -------
    List[SceneObject]
    """

    objects = []

    object_counter = 0

    for class_id in range(num_classes):

        binary = (mask == class_id).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        for label in range(1, num_labels):

            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < min_area:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])

            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])

            component_mask = labels == label

            cx, cy = centroids[label]

            objects.append(

                SceneObject(

                    class_id=class_id,

                    object_id=object_counter,

                    mask=component_mask,

                    area_pixels=area,

                    bbox=(x, y, w, h),

                    centroid=(float(cx), float(cy))

                )

            )

            object_counter += 1

    return objects