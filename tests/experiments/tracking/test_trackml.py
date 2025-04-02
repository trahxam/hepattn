import pytest
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from hepattn.experiments.tracking.data import TrackingDataset
from hepattn.models.matcher import Matcher
from hepattn.models.loss import mask_ce_costs


plt.rcParams["figure.dpi"] = 300


class TestTrackMLEvent:
    @pytest.fixture
    def trackml_event(self):
        input_fields = {
        "hit": [
            "x",
            "y",
            "z",
            "geta",
            "gphi",
        ]
        }

        target_fields = {
            "particle": [
                "pt",
                "eta",
                "phi"
            ],
        }

        dirpath = "/share/rcifdata/maxhart/data/trackml/raw/train/"
        num_samples = -1
        hit_volume_ids = [8]
        particle_min_pt = 1.0
        particle_max_abs_eta = 2.5
        particle_min_num_hits = 3
        event_max_num_particles = 1000

        dataset = TrackingDataset(
            dirpath=dirpath,
            inputs=input_fields,
            targets=target_fields,
            num_samples=num_samples,
            hit_volume_ids=hit_volume_ids,
            particle_min_pt=particle_min_pt,
            particle_max_abs_eta=particle_max_abs_eta,
            particle_min_num_hits=particle_min_num_hits,
            event_max_num_particles=event_max_num_particles,
        )

        return dataset[0]


    def test_trackml_event_masks(self, trackml_event):
        inputs, targets = trackml_event

        particle_valid = targets["particle_valid"]
        particle_hit_mask = targets["particle_hit_valid"]

        # Invalid particle slots should have no hits
        assert torch.all(~particle_hit_mask[~particle_valid.unsqueeze(-1).expand_as(particle_hit_mask)])


    def test_trackml_event_display(self, trackml_event):
        # Quick event display plotted directly from dataloader to verify things look correct

        inputs, targets = trackml_event

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(8, 4)

        batch_idx = 0

        hit_particle_valid = targets["hit_on_valid_particle"][batch_idx]

        for ax_idx, ax_x, ax_y in [(0, "x", "y"),
                                (1, "z", "y")]:
            ax[ax_idx].scatter(inputs[f"hit_{ax_x}"][batch_idx][hit_particle_valid],
                            inputs[f"hit_{ax_y}"][batch_idx][hit_particle_valid],
                            color="black", s=1.0, alpha=0.5)

            for i in range(targets["particle_valid"][batch_idx].shape[-1]):
                if not targets["particle_valid"][batch_idx][i]:
                    continue
                
                if not targets["particle_pt"][batch_idx][i] > 1.0:
                    continue

                particle_mask = targets["particle_hit_valid"][batch_idx][i]
                ax[ax_idx].plot(inputs[f"hit_{ax_x}"][batch_idx][particle_mask],
                                inputs[f"hit_{ax_y}"][batch_idx][particle_mask], alpha=0.5)

        fig.savefig(Path("tests/outputs/trackml/trackml_event.png"))


    def test_trackml_matcher(self, trackml_event):
        # Setup the matcher
        matcher = Matcher(
            default_solver="scipy",
            adaptive_solver=False,
        )





