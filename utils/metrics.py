"""
utils/metrics.py

Метрики качества сегментации.
"""

import numpy as np
import torch


def compute_iou(pred, target, num_classes):
    """
    Вычисление IoU для каждого класса.
    """

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    ious = []

    for cls in range(num_classes):

        pred_mask = pred == cls
        true_mask = target == cls

        intersection = np.logical_and(
            pred_mask,
            true_mask
        ).sum()

        union = np.logical_or(
            pred_mask,
            true_mask
        ).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)

    return ious