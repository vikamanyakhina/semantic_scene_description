"""
models/losses.py

Функции потерь для семантической сегментации.

Используются:

- Dice Loss
- CrossEntropy Loss
- Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss для многоклассовой сегментации.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        num_classes = logits.shape[1]

        probs = torch.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(
            targets,
            num_classes=num_classes
        )

        targets_one_hot = targets_one_hot.permute(
            0, 3, 1, 2
        ).float()

        dims = (0, 2, 3)

        intersection = torch.sum(
            probs * targets_one_hot,
            dims
        )

        cardinality = torch.sum(
            probs + targets_one_hot,
            dims
        )

        dice = (

            2.0 * intersection + self.smooth

        ) / (

            cardinality + self.smooth

        )

        loss = 1.0 - dice.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss =
        CrossEntropy +
        Dice Loss
    """

    def __init__(
            self,
            ce_weight=1.0,
            dice_weight=1.0
    ):

        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.dice = DiceLoss()

        self.ce_weight = ce_weight

        self.dice_weight = dice_weight

    def forward(
            self,
            logits,
            targets
    ):

        ce_loss = self.ce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        total_loss = (

            self.ce_weight * ce_loss +

            self.dice_weight * dice_loss

        )

        return total_loss