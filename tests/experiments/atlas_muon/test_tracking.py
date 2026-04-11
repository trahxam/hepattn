from reconstruct_anything.experiments.atlas_muon import main

from ..utils import run_test  # noqa: TID252


def test_tracking():
    run_test(main, "tests/experiments/atlas_muon/test_tracking.yaml")
