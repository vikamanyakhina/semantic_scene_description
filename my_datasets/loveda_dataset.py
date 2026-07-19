"""
datasets/loveda_dataset.py

Dataset для работы с LoveDA.
Поддерживает загрузку RGB-изображений и дополнительных текстурных признаков.
"""

from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import config


class LoveDADataset(Dataset):

    def __init__(
            self,
            root_dir,
            split="Train",
            texture=None,
            transform=None,
            use_texture=False,
            texture_type=None, 
            texture_dir=None
          ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.texture = texture
        self.transform = transform
        self.use_texture = use_texture
        self.texture_type = texture_type
        self.texture_dir = texture_dir


        self.samples = []

        for area in ["Urban", "Rural"]:

            image_folder = (
                self.root_dir /
                split /
                area /
                "images_png"
            )

            mask_folder = (
                self.root_dir /
                split /
                area /
                "masks_png"
            )

            if not image_folder.exists():
                continue

            image_paths = sorted(image_folder.glob("*.png"))

            for image_path in image_paths:

                sample = {

                    "image": image_path,

                    "mask": mask_folder / image_path.name,

                    "texture": None

                }

                if texture is not None:

                    sample["texture"] = (

                        self.root_dir /
                        split /
                        area /
                        "textures" /
                        texture /
                        image_path.name

                    )

                self.samples.append(sample)

        print(f"{split}: найдено {len(self.samples)} изображений.")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

      sample = self.samples[idx]

      image = Image.open(sample["image"]).convert("RGB")
      mask = Image.open(sample["mask"])

      # ----------------------------------------
      # Масштабируем изображение
      # ----------------------------------------

      image = image.resize(
      (config.IMAGE_SIZE, config.IMAGE_SIZE),
      Image.BILINEAR
      )

      mask = mask.resize(
      (config.IMAGE_SIZE, config.IMAGE_SIZE),
      Image.NEAREST
      )

      image = np.array(image)

      mask = np.array(mask)

      return image, mask