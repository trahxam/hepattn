from hepattn.tasks.base import REGRESSION_LOSS_FNS, RegressionLossType, Task
from hepattn.tasks.classification import ClassificationTask, ObjectClassificationTask
from hepattn.tasks.hit_filter import HitFilterTask
from hepattn.tasks.mask import ObjectHitMaskTask
from hepattn.tasks.regression import (
    GaussianRegressionTask,
    ObjectHitRegressionTask,
    ObjectRegressionTask,
    RegressionTask,
)

__all__ = [
    "REGRESSION_LOSS_FNS",
    "ClassificationTask",
    "GaussianRegressionTask",
    "HitFilterTask",
    "ObjectClassificationTask",
    "ObjectHitMaskTask",
    "ObjectHitRegressionTask",
    "ObjectRegressionTask",
    "RegressionLossType",
    "RegressionTask",
    "Task",
]
