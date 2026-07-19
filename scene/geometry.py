"""
scene/geometry.py

Геометрический анализ объектов сцены.
"""

from typing import List


def compute_area_percent(area_pixels: int,
                         image_shape):
    """
    Процент площади изображения.
    """

    h, w = image_shape[:2]

    total = h * w

    return area_pixels / total * 100


def compute_position(centroid,
                     image_shape):
    """
    Определение положения объекта
    по центру масс.
    """

    h, w = image_shape[:2]

    x, y = centroid

    # ---------- по горизонтали ----------

    if x < w / 3:
        horizontal = "left"

    elif x < 2 * w / 3:
        horizontal = "center"

    else:
        horizontal = "right"

    # ---------- по вертикали ----------

    if y < h / 3:
        vertical = "top"

    elif y < 2 * h / 3:
        vertical = "center"

    else:
        vertical = "bottom"

    if horizontal == "center" and vertical == "center":
        return "center"

    return f"{horizontal}_{vertical}"


def touches_border(bbox,
                   image_shape):
    """
    Касается ли объект границы изображения.
    """

    h, w = image_shape[:2]

    x, y, bw, bh = bbox

    if x == 0:
        return True

    if y == 0:
        return True

    if x + bw >= w - 1:
        return True

    if y + bh >= h - 1:
        return True

    return False