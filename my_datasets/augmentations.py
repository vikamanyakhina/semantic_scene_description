"""
my_datasets/augmentations.py

Аугментации для обучения модели.
Используются только для обучающей выборки.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation(image_size):

    return A.Compose([

        A.Resize(image_size, image_size),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(

            shift_limit=0.05,

            scale_limit=0.10,

            rotate_limit=20,

            border_mode=0,

            p=0.5

        ),

        A.RandomBrightnessContrast(

            brightness_limit=0.2,

            contrast_limit=0.2,

            p=0.5

        ),

        A.GaussNoise(

            std_range=(0.01, 0.05),

            p=0.2

        ),

        A.Normalize(),

        ToTensorV2()

    ])


def get_val_augmentation(image_size):

    return A.Compose([

        A.Resize(image_size, image_size),

        A.Normalize(),

        ToTensorV2()

    ])