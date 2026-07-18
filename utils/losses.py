"""
utils/losses.py

Функции потерь.
"""

import torch.nn as nn


def get_loss():

    """
    CrossEntropyLoss для многоклассовой сегментации.
    """

    return nn.CrossEntropyLoss()