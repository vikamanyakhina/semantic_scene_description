"""
scene/description.py

Формирование текстового описания сцены.
"""

from collections import defaultdict

from geometry import (
    compute_area_percent,
    compute_position,
    compute_zones,
    touches_border,
    POSITION_NAMES
)

CLASS_NAMES = {

    0: "фон",
    1: "здания",
    2: "дороги",
    3: "водные объекты",
    4: "растительность",
    5: "сельскохозяйственные территории",
    6: "лес",
    7: "прочие объекты"
}


def describe_scene(objects, image_shape):

    grouped = defaultdict(list)

    for obj in objects:
        grouped[obj.class_id].append(obj)

    text = []

    text.append("На изображении обнаружены следующие области:\n")

    # сортировка по площади
    ordered = []

    for class_id, class_objects in grouped.items():

        total = sum(o.area_pixels for o in class_objects)

        ordered.append((class_id, total, class_objects))

    ordered.sort(key=lambda x: x[1], reverse=True)

    for class_id, _, class_objects in ordered:

        total_area = sum(
            obj.area_pixels
            for obj in class_objects
        )

        percent = compute_area_percent(
            total_area,
            image_shape
        )

        largest = max(
            class_objects,
            key=lambda x: x.area_pixels
        )

        position = POSITION_NAMES[
            compute_position(
                largest.centroid,
                image_shape
            )
        ]

        count = len(class_objects)

        if count == 1:

            sentence = (
                f"- {CLASS_NAMES[class_id].capitalize()} "
                f"занимают {percent:.1f}% площади изображения "
                f"и расположены {position}."
            )

        else:

            sentence = (
                f"- {CLASS_NAMES[class_id].capitalize()} "
                f"представлены {count} отдельными объектами, "
                f"занимают {percent:.1f}% площади изображения, "
                f"крупнейшая область расположена {position}."
            )

        text.append(sentence)

    return "\n".join(text)