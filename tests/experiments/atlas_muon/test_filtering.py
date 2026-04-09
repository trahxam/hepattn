from hepattn.experiments.atlas_muon import main

from ..utils import run_test  # noqa: TID252


def test_filtering():
    run_test(main, "tests/experiments/atlas_muon/test_filtering.yaml")
