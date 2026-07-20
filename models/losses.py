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

class FocalLoss(nn.Module):

    def __init__(
            self,
            alpha=1.0,
            gamma=2.0
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
            self,
            logits,
            targets
    ):

        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce)

        focal = self.alpha * (1 - pt) ** self.gamma * ce

        return focal.mean()

class CombinedLoss(nn.Module):

    def __init__(
            self,
            ce_weight=0.5,
            dice_weight=0.5,
            focal_weight=0.3
    ):

        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.dice = DiceLoss()

        self.focal = FocalLoss()

        self.ce_weight = ce_weight

        self.dice_weight = dice_weight

        self.focal_weight = focal_weight

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

        focal_loss = self.focal(
            logits,
            targets
        )

        total = (

            self.ce_weight * ce_loss +

            self.dice_weight * dice_loss +

            self.focal_weight * focal_loss

        )

        return total