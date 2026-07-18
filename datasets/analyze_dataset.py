from pathlib import Path

import numpy as np
from PIL import Image

dataset_path = Path("/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA/Train")

classes = set()

for area in ["Urban", "Rural"]:

    mask_folder = dataset_path / area / "masks_png"

    for mask_path in mask_folder.glob("*.png"):

        mask = np.array(Image.open(mask_path))

        classes.update(np.unique(mask))

print("Все найденные классы:")

print(sorted(classes))