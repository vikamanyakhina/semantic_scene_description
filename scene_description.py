"""
scene_description.py

Формирование семантического описания наблюдаемой сцены
на основе результатов семантической сегментации.
"""

from pathlib import Path
import json
import random
import numpy as np

import config


class SceneDescription:

    def __init__(self, output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.class_names = config.CLASS_NAMES

    # ---------------------------------------------------------

    def calculate_percentages(self, prediction):

        prediction = np.asarray(prediction)

        total_pixels = prediction.size

        percentages = {}

        for class_id, class_name in enumerate(self.class_names):

            pixels = np.sum(prediction == class_id)

            percentages[class_name] = round(
                pixels * 100.0 / total_pixels,
                2
            )

        return percentages

    # ---------------------------------------------------------

    def dominant_class(self, percentages):

        return max(
            percentages,
            key=percentages.get
        )

    # ---------------------------------------------------------

    def scene_type(self, percentages):

        building = percentages["Building"]
        road = percentages["Road"]
        forest = percentages["Forest"]
        agriculture = percentages["Agricultural"]
        water = percentages["Water"]
        urban_green = percentages["Urban Green"]

        urban = building + road

        nature = forest + agriculture + urban_green

        if urban >= 45:
            return "Городская"

        if forest >= 45:
            return "Лесная"

        if water >= 40:
            return "Прибрежная"

        if nature >= 55:
            return "Природная"

        return "Смешанная"

    # ---------------------------------------------------------

    def sentence(self, obj, percent):

        variants = [

            f"{obj} занимает {percent:.2f} % площади изображения.",

            f"Доля объекта «{obj}» составляет {percent:.2f} %.",

            f"{obj} покрывает {percent:.2f} % исследуемой территории.",

            f"{obj} представлена на {percent:.2f} % площади сцены.",

            f"Объект «{obj}» занимает около {percent:.2f} % территории."

        ]

        return random.choice(variants)

    # ---------------------------------------------------------

    def build_description(self, percentages):

        dominant = self.dominant_class(percentages)

        scene = self.scene_type(percentages)

        text = []

        text.append(
            f"Тип сцены: {scene}."
        )

        text.append("")

        text.append(

            f"Доминирующим объектом является "

            f"«{dominant}», "

            f"занимающий {percentages[dominant]:.2f} % площади."

        )

        text.append("")

        sorted_objects = sorted(

            percentages.items(),

            key=lambda x: x[1],

            reverse=True

        )

        for obj, value in sorted_objects:

            if value < 1:

                continue

            text.append(

                self.sentence(obj, value)

            )

        text.append("")

        if scene == "Городская":

            text.append(

                "Вывод: наблюдаемая сцена относится "
                "к городской территории с развитой "
                "застройкой и дорожной сетью."

            )

        elif scene == "Лесная":

            text.append(

                "Вывод: наблюдаемая сцена имеет "
                "преимущественно лесной характер."

            )

        elif scene == "Природная":

            text.append(

                "Вывод: наблюдаемая сцена имеет "
                "преимущественно природный характер."

            )

        elif scene == "Прибрежная":

            text.append(

                "Вывод: на изображении преобладают "
                "водные объекты."

            )

        else:

            text.append(

                "Вывод: наблюдаемая сцена является "
                "смешанной и содержит несколько типов "
                "объектов."

            )

        return "\n".join(text)

    # ---------------------------------------------------------

    def save_json(
            self,
            percentages,
            description
    ):

        scene = {

            "scene_type":

                self.scene_type(percentages),

            "dominant_class":

                self.dominant_class(percentages),

            "objects":

                percentages,

            "description":

                description

        }

        with open(

                self.output_dir /

                "scene.json",

                "w",

                encoding="utf-8"

        ) as f:

            json.dump(

                scene,

                f,

                ensure_ascii=False,

                indent=4

            )

    # ---------------------------------------------------------

    def save_txt(self, description):

        with open(

                self.output_dir /

                "scene.txt",

                "w",

                encoding="utf-8"

        ) as f:

            f.write(description)

    # ---------------------------------------------------------

    def describe(self, prediction):

        percentages = self.calculate_percentages(
            prediction
        )

        description = self.build_description(
            percentages
        )

        self.save_json(
            percentages,
            description
        )

        self.save_txt(
            description
        )

        return description