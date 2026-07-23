"""
import numpy as np
from PIL import Image

mask = np.array(Image.open("/content/drive/MyDrive/MyProject/outputs/RGB/inference/5180/prediction.png"))

print("Shape:", mask.shape)
print("Unique:", np.unique(mask))
"""
"""
from PIL import Image
import numpy as np
from pathlib import Path
import config


mask = np.array(
    Image.open(
        Path(config.DATASET_PATH)
        / "Train"
        / "Urban"
        / "masks_png"
        / sorted((Path(config.DATASET_PATH) / "Train" / "Urban" / "masks_png").glob("*.png"))[0].name
    )
)

print(np.unique(mask))
"""
from pathlib import Path
from PIL import Image
import numpy as np
import config


classes = set()

for mask in Path(config.DATASET_PATH).rglob("masks_png/*.png"):
    img = np.array(Image.open(mask))
    classes.update(np.unique(img))

print(sorted(classes))