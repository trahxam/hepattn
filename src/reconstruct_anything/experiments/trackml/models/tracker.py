import torch

from reconstruct_anything.models import ModelWrapper


class TrackMLTracker(ModelWrapper):
    def log_custom_metrics(self, preds, targets, stage):
        query_mask = targets.get("query_mask")
        if query_mask is not None:
            query_mask = query_mask.bool()

        # log intermediate layer mask predictions
        for layer_name, layer_preds in preds.items():
            if "track_hit_valid" not in layer_preds:
                continue
            mask = layer_preds["track_hit_valid"]["track_hit_valid"]
            if mask is not None:
                num_valid = mask.sum(-1).float()
                frac_valid = num_valid / mask.shape[-1]

                if query_mask is None:
                    self.log(f"{stage}/{layer_name}_avg_num_valid_hits", torch.mean(num_valid), sync_dist=True)
                    self.log(f"{stage}/{layer_name}_avg_frac_valid_hits", torch.mean(frac_valid), sync_dist=True)
                else:
                    valid_num_valid = num_valid[query_mask]
                    valid_frac_valid = frac_valid[query_mask]
                    if valid_num_valid.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_num_valid_hits", valid_num_valid.mean(), sync_dist=True)
                    if valid_frac_valid.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_frac_valid_hits", valid_frac_valid.mean(), sync_dist=True)

        preds = preds["final"]

        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        if query_mask is not None:
            pred_valid = pred_valid & query_mask

        pred_hit_masks = preds["track_hit_valid"]["track_hit_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets["particle_hit_valid"] & true_valid.unsqueeze(-1)

        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)
        hit_p = pred_hit_masks.sum(-1)
        hit_t = true_hit_masks.sum(-1)

        for wp in [0.5, 0.75, 1.0]:
            both_valid = true_valid & pred_valid

            effs = (hit_tp / hit_t >= wp) & both_valid
            purs = (hit_tp / hit_p >= wp) & both_valid

            roi_effs = effs.float().sum(-1) / true_valid.float().sum(-1)
            roi_purs = purs.float().sum(-1) / pred_valid.float().sum(-1)

            mean_eff = roi_effs.nanmean()
            mean_pur = roi_purs.nanmean()

            self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

        pred_num = pred_valid.sum(-1)

        nh_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()

        self.log(f"{stage}/nh_per_particle", torch.mean(nh_per_true.float()), sync_dist=True)
        self.log(f"{stage}/nh_per_track", torch.mean(nh_per_pred.float()), sync_dist=True)

        self.log(f"{stage}/num_tracks", torch.mean(pred_num.float()), sync_dist=True)
        true_num = true_valid.sum(-1)
        self.log(f"{stage}/num_particles", torch.mean(true_num.float()), sync_dist=True)

        num_hits_total = float(pred_hit_masks.shape[-1])
        num_hits_valid = float(true_hit_masks.sum())
        num_hits_noise = num_hits_total - num_hits_valid
        self.log(f"{stage}/num_hits", num_hits_total, sync_dist=True)
        self.log(f"{stage}/num_hits_valid", num_hits_valid, sync_dist=True)
        self.log(f"{stage}/num_hits_noise", num_hits_noise, sync_dist=True)
