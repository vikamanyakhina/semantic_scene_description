from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from textures.pygabor import compute_gabor

image_path = Path("/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA/Train/Urban/images_png/1383.png")

image = np.array(Image.open(image_path).convert("RGB"))

gabor = compute_gabor(image)

plt.figure(figsize=(12,6))

plt.subplot(121)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(122)
plt.imshow(gabor, cmap="gray")
plt.title("Gabor")
plt.axis("off")

plt.show()