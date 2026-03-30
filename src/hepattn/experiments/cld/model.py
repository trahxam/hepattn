import random

import matplotlib.pyplot as plt
import torch
from torch import nn

from hepattn.experiments.cld.event_display import plot_cld_event
from hepattn.models.wrapper import ModelWrapper


class CLDReconstructor(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
        pretrained_ckpt_path: str | None = None,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl, pretrained_ckpt_path)
        self._val_display_data = None
        self._val_display_figs = None

    def log_custom_metrics(self, inputs, preds, targets, stage):
        # Just log predictions from the final layer
        preds = preds["final"]

        if self.model.unified_decoding:
            for input_name in self.model.input_names:
                device = inputs[input_name + "_valid"].device
                mask = torch.cat([torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device) for i in self.model.input_names], dim=-1)

                preds[f"flow_{input_name}_assignment"] = {
                    f"flow_{input_name}_valid": preds["flow_key_assignment"]["flow_key_valid"][:,:,mask]
                }                

        hits = [
            "vtxd",
            "trkr",
            "sihit",
            "ecal",
            "hcal",
            "vtb",
            "muon",
        ]

        if "flow_valid" in preds:
            pred_valid = preds["flow_valid"]["flow_valid"]
            true_valid = targets["particle_valid"]

            pred_num = pred_valid.sum(-1)
            true_num = true_valid.sum(-1)

            self.log(f"{stage}/num_flows", torch.mean(pred_num.float()))
            self.log(f"{stage}/num_parts", torch.mean(true_num.float()))

        for hit in hits:
            if f"flow_{hit}_assignment" not in preds:
                continue

            pred_hit_masks = preds[f"flow_{hit}_assignment"][f"flow_{hit}_valid"]
            true_hit_masks = targets[f"particle_{hit}_valid"]
            
            # Only count it is as valid if it has hits
            pred_valid = preds["flow_valid"]["flow_valid"] & (pred_hit_masks.sum(-1) > 0)
            true_valid = targets["particle_valid"] & (true_hit_masks.sum(-1) > 0)

            # Mask out hits that are not on a valid object slot
            pred_hit_masks &= pred_valid.unsqueeze(-1)
            true_hit_masks &= true_valid.unsqueeze(-1)

            # Calculate the true/false positive rates between the predicted and true masks
            # Number of hits that were correctly assigned to the flow
            hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)

            # Number of predicted hits on the flow
            hit_p = pred_hit_masks.sum(-1)

            # True number of hits on the particle
            hit_t = true_hit_masks.sum(-1)

            # Calculate the efficiency and purity at differnt matching working points
            for wp in [0.5, 0.75, 1.0]:
                both_valid = true_valid & pred_valid

                # Whether a truth object is efficient
                effs = ((hit_tp / hit_t) >= wp) & both_valid

                # Whether a pred object is pure / not fake
                purs = ((hit_tp / hit_p) >= wp) & both_valid

                # Calculate the event efficiency / purity
                eff = effs.float().sum(-1) / true_valid.float().sum(-1)
                pur = purs.float().sum(-1) / pred_valid.float().sum(-1)

                self.log(f"{stage}/p{wp}_{hit}_eff", eff.mean())
                self.log(f"{stage}/p{wp}_{hit}_pur", pur.mean())

                # Log some counting info
                pred_num = pred_valid.sum(-1)
                true_num = true_valid.sum(-1)

                num_hits_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()
                num_hits_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()

                self.log(f"{stage}/num_{hit}_per_flow", torch.mean(num_hits_per_pred.float()))
                self.log(f"{stage}/num_{hit}_per_part", torch.mean(num_hits_per_true.float()))


    _N_VAL_DISPLAY = 5

    def on_validation_epoch_start(self):
        self._val_display_data = []

    def validation_step(self, batch):
        inputs, targets = batch
        self._propagate_global_step()
        outputs = self.model(inputs)
        losses, targets = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")
        preds = self.model.predict(outputs)
        self.log_metrics(inputs, preds, targets, "val")

        # Collect one random event per batch until we have _N_VAL_DISPLAY events
        if self.trainer.is_global_zero and len(self._val_display_data) < self._N_VAL_DISPLAY:
            batch_size = next(iter(inputs.values())).shape[0]
            event_idx = random.randint(0, batch_size - 1)
            self._val_display_data.append((
                {k: v[event_idx : event_idx + 1].cpu() for k, v in inputs.items()},
                self._flatten_preds_for_display(preds, event_idx),
                {k: v[event_idx : event_idx + 1].cpu() for k, v in targets.items()},
            ))

        return {"loss": total_loss}

    def _flatten_preds_for_display(self, preds, event_idx):
        """Extract final-layer flow predictions into a flat dict for plot_cld_event."""
        result = {}
        final = preds.get("final", {})
        if "flow_valid" in final:
            result["flow_valid"] = final["flow_valid"]["flow_valid"][event_idx : event_idx + 1].cpu()
        for hit in ["vtxd", "trkr", "ecal", "hcal", "muon"]:
            key = f"flow_{hit}_assignment"
            if key in final:
                val = final[key].get(f"flow_{hit}_valid")
                if val is not None:
                    result[f"flow_{hit}_valid"] = val[event_idx : event_idx + 1].cpu()
        return result

    def on_validation_epoch_end(self):
        # Close any figures from the previous epoch that were not saved by a checkpoint
        if self._val_display_figs:
            for event_figs in self._val_display_figs:
                for fig in event_figs.values():
                    plt.close(fig)
            self._val_display_figs = None

        if not self.trainer.is_global_zero or not self._val_display_data:
            return

        all_display_names = ["vtxd", "trkr", "ecal", "hcal", "muon"]

        all_event_figs = []
        for inputs, flat_preds, targets in self._val_display_data:
            data = {**inputs, **targets, **flat_preds}

            available = [n for n in all_display_names if f"{n}_pos.x" in data]
            if not available:
                all_event_figs.append({})
                continue

            axes_spec = [
                {"x": "pos.x", "y": "pos.y", "input_names": available},
                {"x": "pos.z", "y": "pos.y", "input_names": available},
            ]

            event_figs = {}
            for obj_name, title in [("particle", "Truth"), ("pandora", "Pandora"), ("flow", "Model")]:
                if f"{obj_name}_valid" not in data:
                    continue
                if not any(f"{obj_name}_{n}_valid" in data for n in available):
                    continue
                try:
                    fig = plot_cld_event(data, axes_spec, obj_name, batch_idx=0)
                    fig.suptitle(f"{title} — Epoch {self.current_epoch}", y=1.01)
                    event_figs[obj_name] = fig
                except Exception as e:
                    print(f"[CLDReconstructor] Could not generate {title} event display: {e}")
            all_event_figs.append(event_figs)

        self._val_display_figs = all_event_figs if any(all_event_figs) else None
        self._val_display_data = []
