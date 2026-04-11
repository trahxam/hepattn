from reconstruct_anything.experiments.trackml import main

from ..utils import run_test  # noqa: TID252


def test_filtering():
    run_test(main, "tests/experiments/trackml/test_filtering.yaml")
