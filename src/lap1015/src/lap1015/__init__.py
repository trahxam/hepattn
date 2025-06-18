from __future__ import annotations
import os
from functools import partial

from ._core import linear_sum_assignment

os.environ["OMP_NUM_THREADS"] = "2"

lap_late = partial(linear_sum_assignment, omp=False, eps=False)
lap_early = partial(linear_sum_assignment, omp=True, eps=True)

__all__ = ["linear_sum_assignment", "lap_late", "lap_early"]
