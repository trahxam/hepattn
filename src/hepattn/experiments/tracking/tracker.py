import torch
import torch.nn as nn

from hepattn.models.wrapper import ModelWrapper


class Tracker(ModelWrapper):
    def __init__(
            self,
            model: nn.Module,
            lrs_config: dict,
            optimizer: str = "AdamW",
            mtl: bool = False,
        ):
        super().__init__(model, lrs_config, optimizer, mtl)

    def log_compound_metrics(self, preds, targets, stage):
        # Just log predictions from the final layer
        preds = preds["final"]

        # First log metrics that depend on outputs from multiple tasks
        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        # Set the masks of any track slots that are not used as null
        pred_hit_masks = preds["track_hit_valid"]["track_hit_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets["particle_hit_valid"]

        # Calculate the true/false positive rates between the predicted and true masks
        # Number of hits that were correctly assigned to the track
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)

        # Number of predicted hits on the track
        hit_p = pred_hit_masks.sum(-1)

        # True number of hits on the track
        hit_t = true_hit_masks.sum(-1)

        # Calculate the efficiency and purity at differnt matching working points
        for wp in [0.5, 0.75, 1.0]:
            both_valid = true_valid & pred_valid

            effs = (hit_tp / hit_t >= wp) & both_valid
            purs = (hit_tp / hit_p >= wp) & both_valid

            roi_effs = effs.float().sum(-1) / true_valid.float().sum(-1)
            roi_purs = purs.float().sum(-1) / pred_valid.float().sum(-1)

            mean_eff = roi_effs.nanmean()
            mean_pur = roi_purs.nanmean()

            self.log(f"{stage}_p{wp}_eff", mean_eff)
            self.log(f"{stage}_p{wp}_pur", mean_pur)

        # Log ratio of number of predicted to number of true tracks
        pred_num_tracks = pred_valid.sum(-1)
        true_num_tracks = true_valid.sum(-1)

        pred_num_hits_per_track = pred_hit_masks.sum(-1).float()[pred_valid].mean()
        true_num_hits_per_track = true_hit_masks.sum(-1).float()[true_valid].mean()

        pred_num_hits_per_track_all = pred_hit_masks.sum(-1).float().mean()
        true_num_hits_per_track_all = true_hit_masks.sum(-1).float().mean()

        self.log(f"{stage}_pred_num_hits_per_track", torch.mean(pred_num_hits_per_track.float()))
        self.log(f"{stage}_true_num_hits_per_track", torch.mean(true_num_hits_per_track.float()))

        self.log(f"{stage}_pred_num_hits_per_track_all", torch.mean(pred_num_hits_per_track_all.float()))
        self.log(f"{stage}_true_num_hits_per_track_all", torch.mean(true_num_hits_per_track_all.float()))

        self.log(f"{stage}_num_pred_tracks", torch.mean(pred_num_tracks.float()))
        self.log(f"{stage}_num_true_tracks", torch.mean(true_num_tracks.float()))
        self.log(f"{stage}_ratio_num_pred_true_tracks", torch.mean(pred_num_tracks / true_num_tracks))