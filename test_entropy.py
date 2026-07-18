from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from textures.entropy import compute_entropy

image_path = Path("/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA/Train/Urban/images_png/1383.png")

image = np.array(Image.open(image_path).convert("RGB"))

entropy_map = compute_entropy(image)

plt.figure(figsize=(12,6))

plt.subplot(121)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(122)
plt.imshow(entropy_map, cmap="gray")
plt.title("Entropy")
plt.axis("off")

plt.show()