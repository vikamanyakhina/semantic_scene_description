import numpy as np
from PIL import Image

mask = np.array(Image.open("/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA/Train/Rural/masks_png/564.png"))

print("Уникальные значения:", np.unique(mask))
