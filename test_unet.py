"""
Проверка корректности реализации U-Net.
"""

import torch

from models.unet import UNet


# RGB
model = UNet(
    in_channels=3,
    num_classes=8
)

x = torch.randn(1, 3, 512, 512)

y = model(x)

print("RGB")

print(x.shape)

print(y.shape)


# RGB + Texture

model = UNet(
    in_channels=4,
    num_classes=8
)

x = torch.randn(1, 4, 512, 512)

y = model(x)

print()

print("RGB + Texture")

print(x.shape)

print(y.shape)