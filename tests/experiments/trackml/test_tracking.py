from reconstruct_anything.experiments.trackml import main

from ..utils import run_test  # noqa: TID252


def test_tracking():
    run_test(main, "tests/experiments/trackml/test_tracking.yaml")


def test_tracking_old_sort():
    run_test(main, "tests/experiments/trackml/test_tracking_old_sort.yaml")


def test_tracking_dynamic_queries():
    run_test(main, "tests/experiments/trackml/test_dynamic_queries.yaml")
