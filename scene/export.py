"""
scene/export.py

Сохранение результатов анализа сцены.
"""

import json
from pathlib import Path

from geometry import (
    compute_area_percent,
    compute_position,
    compute_zones,
    touches_border,
    compute_aspect_ratio
)

CLASS_NAMES = {
    0: "background",
    1: "building",
    2: "road",
    3: "water",
    4: "vegetation",
    5: "agricultural",
    6: "forest",
    7: "other"
}


def export_json(objects,
                image_shape,
                save_path):
    """
    Сохранение структурированного описания сцены.
    """

    save_path = Path(save_path)

    result = {
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        "objects": []
    }

    for obj in objects:

        result["objects"].append({

            "id": obj.object_id,

            "class": CLASS_NAMES[obj.class_id],

            "area_pixels": obj.area_pixels,

            "area_percent":
                round(
                    compute_area_percent(
                        obj.area_pixels,
                        image_shape
                    ),
                    2
                ),

            "position":
                compute_position(
                    obj.centroid,
                    image_shape
                ),

            "zones":
                compute_zones(
                    obj.bbox,
                    image_shape
                ),

            "bbox":
                list(obj.bbox),

            "center":
                [
                    round(obj.centroid[0], 2),
                    round(obj.centroid[1], 2)
                ],

            "touches_border":
                touches_border(
                    obj.bbox,
                    image_shape
                ),

            "aspect_ratio":
                round(
                    compute_aspect_ratio(
                        obj.bbox
                    ),
                    2
                )
        })

    with open(save_path,
              "w",
              encoding="utf-8") as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4
        )


def export_txt(description,
               save_path):
    """
    Сохранение текстового описания.
    """

    save_path = Path(save_path)

    with open(save_path,
              "w",
              encoding="utf-8") as f:

        f.write(description)