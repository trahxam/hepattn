from reconstruct_anything.models import ModelWrapper


class TIDEModel(ModelWrapper):
    def log_custom_metrics(self, preds, targets, stage):
        preds = preds["final"]

        hits = ["pix", "sct"]

        if "pred_valid" not in preds:
            return

        pred_valid = preds["pred_valid"]["pred_valid"]
        true_valid = targets["sudo_valid"]

        for hit in hits:
            if f"pred_{hit}_assignment" not in preds:
                continue

            pred_hit_masks = preds[f"pred_{hit}_assignment"][f"pred_{hit}_valid"] & pred_valid.unsqueeze(-1)
            true_hit_masks = targets[f"sudo_{hit}_valid"]

            # Diagonal hit counts (matched pairs after Hungarian permutation)
            hit_tp = (pred_hit_masks & true_hit_masks).sum(-1).float()
            hit_p = pred_hit_masks.sum(-1).float()
            hit_t = true_hit_masks.sum(-1).float()

            # Full pred x true hit-overlap matrix
            # (B, num_pred, num_hits) @ (B, num_hits, num_true) -> (B, num_pred, num_true)
            hit_tp_full = pred_hit_masks.float() @ true_hit_masks.float().transpose(1, 2)
            hit_eff_full = hit_tp_full / hit_t.unsqueeze(1).clamp(min=1)
            hit_pur_full = hit_tp_full / hit_p.unsqueeze(2).clamp(min=1)

            for wp in [0.5, 0.75, 1.0]:
                both_valid = true_valid & pred_valid

                # Diagonal metrics (matched pairs)
                effs = (hit_tp / hit_t.clamp(min=1) >= wp) & both_valid
                purs = (hit_tp / hit_p.clamp(min=1) >= wp) & both_valid

                # Primary match: designated match for its paired true and quality is good
                primary_match = effs & purs

                # Does this pred match ANY true particle above threshold?
                any_match = ((hit_eff_full >= wp) & (hit_pur_full >= wp) & true_valid.unsqueeze(1)).any(-1)

                # Classify pred tracks
                fakes = pred_valid & ~any_match
                dups = pred_valid & any_match & ~primary_match

                n_pred = pred_valid.float().sum(-1).clamp(min=1)
                n_true = true_valid.float().sum(-1).clamp(min=1)

                roi_effs = effs.float().sum(-1) / n_true
                roi_purs = primary_match.float().sum(-1) / n_pred
                roi_faks = fakes.float().sum(-1) / n_pred
                roi_dups = dups.float().sum(-1) / n_pred

                self.log(f"{stage}_p{wp}_{hit}_eff", roi_effs.nanmean())
                self.log(f"{stage}_p{wp}_{hit}_pur", roi_purs.nanmean())
                self.log(f"{stage}_p{wp}_{hit}_fak", roi_faks.nanmean())
                self.log(f"{stage}_p{wp}_{hit}_dup", roi_dups.nanmean())
