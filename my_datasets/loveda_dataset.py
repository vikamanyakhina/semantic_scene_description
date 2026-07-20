"""
datasets/loveda_dataset.py

Dataset для LoveDA.

Поддерживает:

- RGB
- RGB + Texture
- автоматическое изменение размера
- использование config.py
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
            use_texture=False,
            transform=None
    ):

        self.root_dir = Path(root_dir)

        self.split = split

        self.texture = texture

        self.use_texture = use_texture

        self.transform = transform

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

            for image_path in sorted(image_folder.glob("*.png")):

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

        # ---------------------------------------------------
        # RGB
        # ---------------------------------------------------

        image = Image.open(
            sample["image"]
        ).convert("RGB")

        mask = Image.open(
            sample["mask"]
        )

        # ---------------------------------------------------
        # Resize
        # ---------------------------------------------------

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

        # ---------------------------------------------------
        # Texture
        # ---------------------------------------------------

        if self.use_texture and sample["texture"] is not None:

            texture = Image.open(sample["texture"])

            texture = texture.resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE),
                Image.BILINEAR
            )

            texture = np.array(texture)

            if texture.ndim == 2:

                texture = texture[..., None]

            image = np.concatenate(
                (image, texture),
                axis=2
            )

        # ---------------------------------------------------
        # Normalize
        # ---------------------------------------------------

        if self.transform is not None:

            transformed = self.transform(

                image=image,

                mask=mask

            )

            image = transformed["image"].float()

            mask = transformed["mask"].long()

        else:

            image = image.astype(np.float32) / 255.0

            image = torch.from_numpy(image).permute(2, 0, 1).float()

            mask = torch.from_numpy(mask).long()
       

        return image, mask