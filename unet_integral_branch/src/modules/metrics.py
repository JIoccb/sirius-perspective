from __future__ import annotations

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryJaccardIndex
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError


def get_mask_metrics(threshold: float = 0.5) -> MetricCollection:
    """Binary segmentation metrics computed on probabilities (after sigmoid)."""
    return MetricCollection({
        'f1': BinaryF1Score(threshold=threshold),
        'precision': BinaryPrecision(threshold=threshold),
        'recall': BinaryRecall(threshold=threshold),
        'iou': BinaryJaccardIndex(threshold=threshold),
    })


def get_kps_metrics() -> MetricCollection:
    """Regression metrics for keypoints coordinates."""
    return MetricCollection({
        'mae': MeanAbsoluteError(),
        'mse': MeanSquaredError(),
    })
