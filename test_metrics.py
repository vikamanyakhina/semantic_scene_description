import torch

from utils.metrics import SegmentationMetrics

metrics = SegmentationMetrics(8)

pred = torch.rand(
    2,
    8,
    256,
    256
)

gt = torch.randint(
    0,
    8,
    (2, 256, 256)
)

metrics.update(
    pred,
    gt
)

print(metrics.summary())