from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from textures.lbp import compute_lbp

# Путь к любому изображению
image_path = Path(
    "/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA/Train/Urban/images_png/1383.png"
)

image = np.array(Image.open(image_path).convert("RGB"))

lbp = compute_lbp(image)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(lbp, cmap="gray")
plt.title("LBP")
plt.axis("off")

plt.show()