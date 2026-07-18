"""
create_small_dataset.py

Создание уменьшенной версии датасета LoveDA для проведения экспериментов.

Структура нового датасета полностью повторяет оригинальную.
Изображения выбираются случайным образом с фиксированным random seed,
что делает эксперименты воспроизводимыми.
"""

from pathlib import Path
import random
import shutil


# ------------------------------------------------------------
# Настройки
# ------------------------------------------------------------

SOURCE_DATASET = Path("/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA")
TARGET_DATASET = Path("/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA_small")

RANDOM_SEED = 42

SAMPLES = {
    "Train": 100,
    "Val": 50
}


# ------------------------------------------------------------
# Создание выборки
# ------------------------------------------------------------

random.seed(RANDOM_SEED)


def copy_split(split_name):

    print(f"\n========== {split_name} ==========")

    for area in ["Urban", "Rural"]:

        print(area)

        image_folder = (
            SOURCE_DATASET /
            split_name /
            area /
            "images_png"
        )

        mask_folder = (
            SOURCE_DATASET /
            split_name /
            area /
            "masks_png"
        )

        images = sorted(image_folder.glob("*.png"))

        selected = random.sample(
            images,
            SAMPLES[split_name]
        )

        new_image_folder = (
            TARGET_DATASET /
            split_name /
            area /
            "images_png"
        )

        new_mask_folder = (
            TARGET_DATASET /
            split_name /
            area /
            "masks_png"
        )

        new_image_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        new_mask_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for image_path in selected:

            shutil.copy2(
                image_path,
                new_image_folder / image_path.name
            )

            shutil.copy2(
                mask_folder / image_path.name,
                new_mask_folder / image_path.name
            )

        print(f"Скопировано {len(selected)} изображений.")


def main():

    for split in ["Train", "Val"]:
        copy_split(split)

    print("\nГотово!")
    print(f"Новый датасет находится в:\n{TARGET_DATASET}")


if __name__ == "__main__":
    main()