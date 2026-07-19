"""
utils/metrics.py

Расчет метрик качества для задачи семантической сегментации.

Поддерживаются:

- Pixel Accuracy
- IoU по каждому классу
- Mean IoU
- Dice Score
- Confusion Matrix

Автор: Victoria + ChatGPT
"""

import numpy as np
import torch


class SegmentationMetrics:
    """
    Накопление confusion matrix и вычисление метрик.
    """

    def __init__(self, num_classes: int):

        self.num_classes = num_classes

        self.reset()

    # -------------------------------------------------------

    def reset(self):

        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64
        )

    # -------------------------------------------------------

    @torch.no_grad()
    def update(self,
               prediction: torch.Tensor,
               target: torch.Tensor):
        """
        prediction : BxCxHxW (логиты модели)

        target : BxHxW
        """

        prediction = torch.argmax(
            prediction,
            dim=1
        )

        prediction = prediction.cpu().numpy().reshape(-1)

        target = target.cpu().numpy().reshape(-1)

        mask = (
            target >= 0
        ) & (
            target < self.num_classes
        )

        prediction = prediction[mask]

        target = target[mask]

        cm = np.bincount(
            self.num_classes * target + prediction,
            minlength=self.num_classes ** 2
        )

        cm = cm.reshape(
            self.num_classes,
            self.num_classes
        )

        self.confusion_matrix += cm

    # -------------------------------------------------------

    def pixel_accuracy(self):

        cm = self.confusion_matrix

        return np.diag(cm).sum() / cm.sum()

    # -------------------------------------------------------

    def class_accuracy(self):

        cm = self.confusion_matrix

        acc = np.diag(cm) / (
            cm.sum(axis=1) + 1e-8
        )

        return acc

    # -------------------------------------------------------

    def iou(self):

        cm = self.confusion_matrix

        intersection = np.diag(cm)

        union = (
            cm.sum(axis=1)
            + cm.sum(axis=0)
            - intersection
        )

        iou = intersection / (
            union + 1e-8
        )

        return iou

    # -------------------------------------------------------

    def mean_iou(self):

        return np.nanmean(
            self.iou()
        )

    # -------------------------------------------------------

    def dice(self):

        cm = self.confusion_matrix

        intersection = np.diag(cm)

        dice = (
            2 * intersection
        ) / (
            cm.sum(axis=1)
            + cm.sum(axis=0)
            + 1e-8
        )

        return dice

    # -------------------------------------------------------

    def mean_dice(self):

        return np.nanmean(
            self.dice()
        )

    # -------------------------------------------------------

    def summary(self):
        """
        Возвращает словарь всех метрик.
        """

        return {

            "pixel_accuracy":
                float(
                    self.pixel_accuracy()
                ),

            "mean_iou":
                float(
                    self.mean_iou()
                ),

            "mean_dice":
                float(
                    self.mean_dice()
                ),

            "class_accuracy":
                self.class_accuracy().tolist(),

            "iou":
                self.iou().tolist(),

            "dice":
                self.dice().tolist()
        }