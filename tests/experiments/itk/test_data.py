import pytest
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from hepattn.experiments.itk.data import ITkDataset
from hepattn.experiments.itk.plot_event import plot_itk_event_reconstruction
from hepattn.models.matcher import Matcher
from hepattn.models.loss import mask_ce_costs


plt.rcParams["figure.dpi"] = 300


class TestITkEvent:
    @pytest.fixture
    def itk_event(self):
        input_fields = {
        "pixel": [
            "x",
            "y",
            "z",
            "r",
            "eta",
            "phi",
            "u",
            "v",
        ],

        "strip": [
            "x",
            "y",
            "z",
            "r",
            "eta",
            "phi",
            "u",
            "v",
        ]
        }

        target_fields = {
            "particle": [
                "pt",
                "eta",
                "phi"
            ],
        }

        dirpath = "/share/rcifdata/maxhart/data/itk/val"
        num_events = -1
        hit_volume_ids = [8]
        particle_min_pt = 1.0
        particle_max_abs_eta = 1.0
        particle_min_num_pixels = 3
        particle_min_num_strips = 3
        event_max_num_particles = 2000

        dataset = ITkDataset(
            dirpath=dirpath,
            inputs=input_fields,
            targets=target_fields,
            num_events=num_events,
            hit_volume_ids=hit_volume_ids,
            particle_min_pt=particle_min_pt,
            particle_max_abs_eta=particle_max_abs_eta,
            particle_min_num_pixels=particle_min_num_pixels,
            particle_min_num_strips=particle_min_num_strips,
            event_max_num_particles=event_max_num_particles,
        )

        return dataset[0]


    def test_itk_event_masks(self, itk_event):
        # Some basic sanity checks on the event data
        inputs, targets = itk_event

        # Every valid particle should have a unique particle id

        # Particle id should be a long


        


    def test_itk_event_display(self, itk_event):
        # Plot an event display directly from dataloader to verify things look correct
        inputs, targets = itk_event

        fig = plot_itk_event_reconstruction(inputs, targets)
        fig.savefig(Path("tests/outputs/itk/itk_event.png"))


    def test_itk_matcher(self, itk_event):
        # Setup the matcher
        matcher = Matcher(
            default_solver="scipy",
            adaptive_solver=False,
        )
