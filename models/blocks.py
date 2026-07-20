"""
models/blocks.py

Основные строительные блоки архитектуры U-Net.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Две последовательные свертки 3x3 с BatchNorm и ReLU.
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.double_conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Dropout2d(p=0.10),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Dropout2d(p=0.10),

        )

    def forward(self, x):

        return self.double_conv(x)


class Down(nn.Module):
    """
    Уменьшение пространственного размера изображения.
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.down = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(
                in_channels,
                out_channels
            )

        )

    def forward(self, x):

        return self.down(x)


class Up(nn.Module):
    """
    Увеличение изображения и объединение со skip-соединением.
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(

            x1,

            [

                diff_x // 2,
                diff_x - diff_x // 2,

                diff_y // 2,
                diff_y - diff_y // 2

            ]

        )

        x = torch.cat(
            [x2, x1],
            dim=1
        )

        return self.conv(x)


class OutConv(nn.Module):
    """
    Последняя свертка 1x1.
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        return self.conv(x)