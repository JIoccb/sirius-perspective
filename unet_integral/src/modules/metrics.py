from torchmetrics import MetricCollection, F1Score, Precision, Recall, JaccardIndex, MeanAbsoluteError, MeanSquaredError


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection({
        'f1': F1Score(**kwargs),
        'precision': Precision(**kwargs),
        'recall': Recall(**kwargs),
        'iou': JaccardIndex(**kwargs),
        'mae': MeanAbsoluteError(**kwargs),
        'mse': MeanSquaredError(**kwargs)
    })