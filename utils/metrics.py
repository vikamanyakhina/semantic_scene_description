"""
utils/metrics.py

Метрики качества семантической сегментации.

Поддерживает:

- Pixel Accuracy
- Mean Accuracy
- IoU
- Dice
- Precision
- Recall
- F1-score
- Confusion Matrix
"""

import numpy as np
import torch


class SegmentationMetrics:

    def __init__(self, num_classes):

        self.num_classes = num_classes

        self.reset()

    # ---------------------------------------------------------

    def reset(self):

        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64
        )

    # ---------------------------------------------------------

    @torch.no_grad()
    def update(self, prediction, target):

        prediction = torch.argmax(prediction, dim=1)

        prediction = prediction.cpu().numpy().reshape(-1)

        target = target.cpu().numpy().reshape(-1)

        valid = (
            (target >= 0) &
            (target < self.num_classes)
        )

        prediction = prediction[valid]

        target = target[valid]

        cm = np.bincount(
            self.num_classes * target + prediction,
            minlength=self.num_classes ** 2
        )

        cm = cm.reshape(
            self.num_classes,
            self.num_classes
        )

        self.confusion_matrix += cm

    # ---------------------------------------------------------

    def pixel_accuracy(self):

        cm = self.confusion_matrix

        return np.diag(cm).sum() / (cm.sum() + 1e-8)

    # ---------------------------------------------------------

    def class_accuracy(self):

        cm = self.confusion_matrix

        return np.diag(cm) / (
            cm.sum(axis=1) + 1e-8
        )

    # ---------------------------------------------------------

    def precision(self):

        cm = self.confusion_matrix

        tp = np.diag(cm)

        fp = cm.sum(axis=0) - tp

        return tp / (tp + fp + 1e-8)

    # ---------------------------------------------------------

    def recall(self):

        cm = self.confusion_matrix

        tp = np.diag(cm)

        fn = cm.sum(axis=1) - tp

        return tp / (tp + fn + 1e-8)

    # ---------------------------------------------------------

    def f1(self):

        p = self.precision()

        r = self.recall()

        return 2 * p * r / (p + r + 1e-8)

    # ---------------------------------------------------------

    def iou(self):

        cm = self.confusion_matrix

        tp = np.diag(cm)

        fp = cm.sum(axis=0) - tp

        fn = cm.sum(axis=1) - tp

        return tp / (tp + fp + fn + 1e-8)

    # ---------------------------------------------------------

    def dice(self):

        cm = self.confusion_matrix

        tp = np.diag(cm)

        fp = cm.sum(axis=0) - tp

        fn = cm.sum(axis=1) - tp

        return 2 * tp / (
            2 * tp + fp + fn + 1e-8
        )

    # ---------------------------------------------------------

    def summary(self):

        precision = self.precision()

        recall = self.recall()

        f1 = self.f1()

        iou = self.iou()

        dice = self.dice()

        return {

            "pixel_accuracy": float(self.pixel_accuracy()),

            "mean_accuracy": float(
                np.nanmean(
                    self.class_accuracy()
                )
            ),

            "precision": precision.tolist(),

            "recall": recall.tolist(),

            "f1": f1.tolist(),

            "mean_precision": float(
                np.nanmean(precision)
            ),

            "mean_recall": float(
                np.nanmean(recall)
            ),

            "mean_f1": float(
                np.nanmean(f1)
            ),

            "class_iou": iou.tolist(),

            "mean_iou": float(
                np.nanmean(iou)
            ),

            "class_dice": dice.tolist(),

            "mean_dice": float(
                np.nanmean(dice)
            ),

            "confusion_matrix":
                self.confusion_matrix.tolist()

        }