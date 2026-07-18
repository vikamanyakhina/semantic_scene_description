"""
precompute_textures.py

Предварительное вычисление текстурных признаков для датасета LoveDA.

После запуска будут созданы папки:

Train/
    Urban/
        textures/
            lbp/
            entropy/
            variance/
            gabor/

Аналогично для Rural, Val и Test.

Текстурные карты вычисляются один раз и сохраняются на диск.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from textures.lbp import compute_lbp
from textures.entropy import compute_entropy
from textures.variance import compute_variance
from textures.pygabor import compute_gabor


# ------------------------------------------------------------
# Путь к датасету
# ------------------------------------------------------------

DATASET_PATH = Path("/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA_small")


# ------------------------------------------------------------
# Какие признаки вычислять
# ------------------------------------------------------------

TEXTURE_FUNCTIONS = {
    "lbp": compute_lbp,
    "entropy": compute_entropy,
    "variance": compute_variance,
    "gabor": compute_gabor,
}


# ------------------------------------------------------------
# Основная программа
# ------------------------------------------------------------

def process_split(split_name: str):

    print(f"\n========== {split_name} ==========")

    split_path = DATASET_PATH / split_name

    for area in ["Urban", "Rural"]:

        image_folder = split_path / area / "images_png"

        if not image_folder.exists():
            continue

        print(f"\n{area}")

        image_list = sorted(image_folder.glob("*.png"))

        for texture_name, texture_function in TEXTURE_FUNCTIONS.items():

            save_folder = (
                split_path
                / area
                / "textures"
                / texture_name
            )

            save_folder.mkdir(parents=True, exist_ok=True)

            print(f"  -> {texture_name}")

            for image_path in tqdm(image_list):

                image = np.array(
                    Image.open(image_path).convert("RGB")
                )

                texture = texture_function(image)

                save_path = save_folder / image_path.name

                cv2.imwrite(
                    str(save_path),
                    texture
                )


def main():

    for split in ["Train", "Val", "Test"]:
        process_split(split)

    print("\nВсе текстурные карты успешно сохранены.")


if __name__ == "__main__":
    main()