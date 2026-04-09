from hepattn.tasks.base import REGRESSION_LOSS_FNS, RegressionLossType, Task
from hepattn.tasks.classification import ClassificationTask, ObjectClassificationTask
from hepattn.tasks.hit_filter import HitFilterTask, HitFilterTaskBatched
from hepattn.tasks.incidence import IncidenceBasedRegressionTask, IncidenceRegressionTask
from hepattn.tasks.iou import IoUPredictionTask
from hepattn.tasks.mask import ObjectHitMaskTask
from hepattn.tasks.regression import (
    GaussianRegressionTask,
    ObjectGaussianRegressionTask,
    ObjectHitRegressionTask,
    ObjectRegressionTask,
    RegressionTask,
)

__all__ = [
    "ClassificationTask",
    "GaussianRegressionTask",
    "HitFilterTask",
    "HitFilterTaskBatched",
    "IncidenceBasedRegressionTask",
    "IncidenceRegressionTask",
    "IoUPredictionTask",
    "ObjectClassificationTask",
    "ObjectGaussianRegressionTask",
    "ObjectHitMaskTask",
    "ObjectHitRegressionTask",
    "ObjectRegressionTask",
    "REGRESSION_LOSS_FNS",
    "RegressionLossType",
    "RegressionTask",
    "Task",
]
