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

"""
scene/geometry.py

Геометрический анализ объектов сцены.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class GeometryInfo:
    area_percent: float
    position: str
    zones: List[str]
    touches_border: bool
    aspect_ratio: float


POSITION_NAMES = {
    "left_top": "в левой верхней части изображения",
    "center_top": "в верхней центральной части изображения",
    "right_top": "в правой верхней части изображения",

    "left_center": "в левой части изображения",
    "center": "в центральной части изображения",
    "right_center": "в правой части изображения",

    "left_bottom": "в левой нижней части изображения",
    "center_bottom": "в нижней центральной части изображения",
    "right_bottom": "в правой нижней части изображения"
}


def compute_area_percent(area_pixels, image_shape):

    h, w = image_shape[:2]

    return area_pixels / (h * w) * 100


def compute_position(centroid, image_shape):

    h, w = image_shape[:2]

    x, y = centroid

    # горизонталь
    if x < w / 3:
        h_pos = "left"
    elif x < 2 * w / 3:
        h_pos = "center"
    else:
        h_pos = "right"

    # вертикаль
    if y < h / 3:
        v_pos = "top"
    elif y < 2 * h / 3:
        v_pos = "center"
    else:
        v_pos = "bottom"

    if h_pos == "center" and v_pos == "center":
        return "center"

    return f"{h_pos}_{v_pos}"


def compute_zones(bbox, image_shape):
    """
    Определяет, какие из 9 зон изображения занимает объект.
    """

    h, w = image_shape[:2]

    x, y, bw, bh = bbox

    x1 = x
    x2 = x + bw

    y1 = y
    y2 = y + bh

    zones = []

    cols = [
        (0, w / 3, "left"),
        (w / 3, 2 * w / 3, "center"),
        (2 * w / 3, w, "right")
    ]

    rows = [
        (0, h / 3, "top"),
        (h / 3, 2 * h / 3, "center"),
        (2 * h / 3, h, "bottom")
    ]

    for cx1, cx2, cx_name in cols:

        if x2 < cx1 or x1 > cx2:
            continue

        for cy1, cy2, cy_name in rows:

            if y2 < cy1 or y1 > cy2:
                continue

            if cx_name == "center" and cy_name == "center":
                zones.append("center")
            else:
                zones.append(f"{cx_name}_{cy_name}")

    return sorted(set(zones))


def touches_border(bbox, image_shape):

    h, w = image_shape[:2]

    x, y, bw, bh = bbox

    return (
        x == 0 or
        y == 0 or
        x + bw >= w - 1 or
        y + bh >= h - 1
    )


def compute_aspect_ratio(bbox):

    _, _, bw, bh = bbox

    if bh == 0:
        return 0.0

    return bw / bh