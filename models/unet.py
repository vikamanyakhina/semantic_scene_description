"""
models/unet.py

Облегченная архитектура U-Net (Light U-Net) для семантической сегментации.
Поддерживает RGB (3 канала) и RGB + текстурный признак (4 канала).
"""

import torch.nn as nn

from models.blocks import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):

    def __init__(
            self,
            in_channels=3,
            num_classes=8,
            base_channels=32
    ):
        """
        Parameters
        ----------
        in_channels : int
            Количество входных каналов (3 или 4).

        num_classes : int
            Количество классов сегментации.

        base_channels : int
            Базовое количество каналов.
            32 -> Light U-Net
            64 -> классический U-Net
        """

        super().__init__()

        c = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        # Decoder
        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)

        # Output
        self.outc = OutConv(c, num_classes)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        return self.outc(x)