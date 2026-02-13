import torch
from torch import Tensor
from typing import Literal
import torch.nn.functional as F

from hepattn.models.task import Task
from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, loss_fns
from hepattn.utils.helix import fit_helices


class CLDTask(Task):
    def __init__(
        self,
        name: str,
        dim: int,
        mask_attn: bool = False,
        has_intermediate_loss: bool = True,
        hits_included: list[str] | str = "all",
        hit_mask_attn_thresholds: dict[str, float] | None = None,
        tracker_helix_fit: bool = False,
        helix_fit_warmup_steps: int = 0,
        calo_line_fit: bool = False,
        calo_score_method: str = "sigmoid",
        loss_object_mask: str = "selective",
        hit_loss_weights: dict[str, dict[str, float]] | None = None,
        hit_cost_weights: dict[str, dict[str, float]] | None = None,
        mask_dice_cost_logit_scale: float = 1.0,
        return_embeddings: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.dim = dim
        self.mask_attn = mask_attn
        self.loss_object_mask = loss_object_mask
        self.calo_score_method = calo_score_method
        self.return_embeddings = return_embeddings
        self.hit_loss_weights = hit_loss_weights or {}
        self.hit_cost_weights = hit_cost_weights or {}
        self.mask_dice_cost_logit_scale = float(mask_dice_cost_logit_scale)

        # Which detector subhits will be used
        if hits_included == "all":
            self.hits_included = ["vtxd", "trkr", "ecal", "hcal", "muon"]
        else:
            self.hits_included = hits_included

        # Map of particle class to class index
        self.class_name_to_idx = {
            "null": 0,
            "neutral_hadron": 1,
            "charged_hadron": 2,
            "photon": 3,
            "electron": 4,
            "muon": 5,
        }

        # Maps which hits should be active for each class
        self.class_active_hits = {
            "null": [],
            "neutral_hadron": ["ecal", "hcal"],
            "charged_hadron": ["vtxd", "trkr", "ecal", "hcal"],
            "photon": ["ecal"],
            "electron": ["vtxd", "trkr", "ecal"],
            "muon": ["vtxd", "trkr", "ecal", "hcal", "muon"],
            "charged": ["vtxd", "trkr"],
            "neutral": ["ecal", "hcal"],
        }

        self.hit_active_classes = {}

        for class_name, hits in self.class_active_hits.items():
            for sys in hits:
                if sys not in self.hit_active_classes:
                    self.hit_active_classes[sys] = []

                self.hit_active_classes[sys].append(class_name)

        self.charged_classes = ["charged_hadron", "electron", "muon"]
        self.neutral_classes = ["neutral_hadron", "photon"]

        self.register_buffer("charged_class_idxs", torch.tensor([2, 4, 5], dtype=torch.long))
        self.register_buffer("neutral_class_idxs", torch.tensor([1, 3], dtype=torch.long))

        # Network for particle classification
        self.class_net = Dense(dim, len(self.class_name_to_idx))

        # Networks for producing mask tokens for assignment
        self.hit_mask_nets = torch.nn.ModuleDict({hit: Dense(dim, dim) for hit in self.hits_included})

        # Defines which outputs of forward() need to be permuted during the matching
        self.outputs = (
            ["flow_logit"]
            + [f"flow_{hit}_logit" for hit in self.hits_included]
            + [f"flow_{hit}_prob" for hit in self.hits_included]
        )

        if tracker_helix_fit:
            self.outputs += [
                "flow_regr_helix",
                "flow_helix_fittable",
                "flow_helix_fitted",
                ]

            self.helix_fit_fields = [
                "mom.rinv",
                "mom.eta",
                "mom.phi",
                "vtx.r",
                "vtx.z",
            ]

            self.helix_refine_net = Dense(len(self.helix_fit_fields), len(self.helix_fit_fields))

        if hit_mask_attn_thresholds is None:
            self.hit_mask_attn_thresholds = {
                "vtxd": 0.5,
                "trkr": 0.5,
                "ecal": 0.1,
                "hcal": 0.1,
                "muon": 0.5,
            }
        else:
            self.hit_mask_attn_thresholds = hit_mask_attn_thresholds

        self.tracker_helix_fit = tracker_helix_fit
        self.helix_fit_warmup_steps = int(helix_fit_warmup_steps)
        self.calo_line_fit = calo_line_fit

    def _helix_fit_enabled(self) -> bool:
        if self.helix_fit_warmup_steps <= 0:
            return True
        step = int(getattr(self, "global_step", 0))
        return step >= self.helix_fit_warmup_steps

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs: dict[str, Tensor] = {}

        outputs["flow_logit"] = self.class_net(x["query_embed"])

        if self.return_embeddings:
            outputs["query_embed"] = x["query_embed"]

        for hit, mask_net in self.hit_mask_nets.items():
            # query-side mask embedding
            q = mask_net(x["query_embed"])      # [B, Nq, C]
            k = x[f"{hit}_embed"]               # [B, Nh, C]

            if self.return_embeddings:
                outputs[f"mask_token_{hit}"] = q
                outputs[f"{hit}_embed"] = k

            # assignments logits: [B, Nq, Nh]
            flow_hit_logit = torch.einsum("bnc,bmc->bnm", q, k)

            # Set padding entries to -inf (mask invalid hits)
            flow_hit_logit[~x[f"{hit}_valid"].unsqueeze(-2).expand_as(flow_hit_logit)] = torch.finfo(
                flow_hit_logit.dtype
            ).min

            outputs[f"flow_{hit}_logit"] = flow_hit_logit

            # Tracker-like hits
            if hit in {"vtxd", "trkr", "muon"}:
                outputs[f"flow_{hit}_prob"] = outputs[f"flow_{hit}_logit"].sigmoid()

            # Calo-like hits (use probs, not log-probs, since you threshold later)
            if hit in {"ecal", "hcal"}:
                # Obtain the assignment just doing a sigmoid
                if self.calo_score_method == "sigmoid":
                    outputs[f"flow_{hit}_prob"] = outputs[f"flow_{hit}_logit"].sigmoid()
                # Softmax over the particle dim
                # In this case, p[i,j] = frac of energy on hit j which was from particle i
                elif self.calo_score_method == "particle_softmax":
                    outputs[f"flow_{hit}_prob"] = outputs[f"flow_{hit}_logit"].softmax(-2)
                # Softmax over the particle dim
                # In this case, p[i,j] = frac of energy of particle i which was provided by hit j
                elif self.calo_score_method == "hit_softmax":
                    outputs[f"flow_{hit}_prob"] = outputs[f"flow_{hit}_logit"].softmax(-1)
        
        flow_class_idx = outputs["flow_logit"].argmax(-1)
        flow_charged = torch.isin(flow_class_idx, self.charged_class_idxs)
        
        if self.tracker_helix_fit:
            helix_fit_enabled = self._helix_fit_enabled()
            flow_sihit_prob = torch.cat([outputs[f"flow_{hit}_prob"] for hit in ("vtxd", "trkr")], dim=-1)
            flow_num_sihit = (flow_sihit_prob >= 0.5).sum(-1)
            flow_fittable = (flow_num_sihit >= 6) & (flow_num_sihit <= 24) & flow_charged

            sihit_x, sihit_y, sihit_z = tuple(
                torch.cat(tuple(x[f"{hit}_pos.{c}"] for hit in ("vtxd", "trkr")), dim=-1)
                for c in ("x", "y", "z")
            )

            if helix_fit_enabled:
                # Perform the fit
                radius, phi0, eta, d0, z0, flow_fitted = fit_helices(
                    sihit_x,
                    sihit_y,
                    sihit_z,
                    flow_sihit_prob * (flow_sihit_prob.detach() >= 0.5).type_as(flow_sihit_prob),
                    flow_fittable.detach(),
                )

                # Record the raw helix fit for diagnostics
                ptinv = 1.0 / (0.3 * 2.0 * radius) # Transform from radius to 1/pT

                outputs["flow_helix_mom.rinv"] = ptinv
                outputs["flow_helix_mom.phi"] = phi0
                outputs["flow_helix_mom.eta"] = eta
                outputs["flow_helix_vtx.r"] = d0 # In m as global coords in m
                outputs["flow_helix_vtx.z"] = z0 # In m as global coords in m
            else:
                flow_fitted = torch.zeros_like(flow_fittable, dtype=torch.bool)
                zero = outputs["flow_logit"].new_zeros(flow_fittable.shape)
                outputs["flow_helix_mom.rinv"] = zero
                outputs["flow_helix_mom.phi"] = zero
                outputs["flow_helix_mom.eta"] = zero
                outputs["flow_helix_vtx.r"] = zero
                outputs["flow_helix_vtx.z"] = zero

            outputs["flow_regr_helix"] = torch.stack(
                [outputs[f"flow_helix_{f}"] for f in self.helix_fit_fields],
                dim=-1,
            )

            if helix_fit_enabled and self.helix_refine_net is not None:
                outputs["flow_regr_helix"] = self.helix_refine_net(outputs["flow_regr_helix"])

            outputs["flow_helix_fittable"] = flow_fittable
            outputs["flow_helix_fitted"] = flow_fitted            

        return outputs

    def attn_mask(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.mask_attn:
            return {}

        attn_masks: dict[str, Tensor] = {}
        for hit in self.hits_included:
            thresh = self.hit_mask_attn_thresholds[hit]
            attn_masks[hit] = outputs[f"flow_{hit}_prob"].detach() >= thresh

        return attn_masks

    def affinity(self, outputs: dict[str, Tensor], x: dict[str, Tensor], num_constituents: int) -> Tensor | None:
        batch_size, num_queries = outputs["flow_logit"].shape[:2]
        affinity_logits = torch.full(
            (batch_size, num_queries, num_constituents),
            float("-inf"),
            device=outputs["flow_logit"].device,
            dtype=outputs["flow_logit"].dtype,
        )

        for hit in self.hits_included:
            raw = outputs[f"flow_{hit}_logit"]
            key_is = x[f"key_is_{hit}"].unsqueeze(1).expand_as(affinity_logits)
            affinity_logits[key_is] = torch.maximum(affinity_logits[key_is], raw.flatten())

        return affinity_logits

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs["flow_class_idx"] = outputs["flow_logit"].argmax(dim=-1)
        outputs["flow_valid"] = outputs["flow_class_idx"] != 0

        for class_name, class_idx in self.class_name_to_idx.items():
            outputs[f"flow_is_{class_name}"] = outputs["flow_class_idx"] == class_idx

        outputs["flow_is_charged"] = torch.isin(outputs["flow_class_idx"], self.charged_class_idxs)
        outputs["flow_is_neutral"] = torch.isin(outputs["flow_class_idx"], self.neutral_class_idxs)

        for hit in self.hits_included:
            outputs[f"flow_{hit}_valid"] = outputs[f"flow_{hit}_prob"] >= 0.5 

        return outputs

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        costs: dict[str, Tensor] = {}

        flow_class_logit = outputs["flow_logit"].detach().to(torch.float32)
        logit_null = flow_class_logit[..., 0]
        logit_nonnull = torch.logsumexp(flow_class_logit[..., 1:], dim=-1)
        valid_logit = logit_nonnull - logit_null
        costs["object_bce"] = 1 + cost_fns["object_bce"](valid_logit, targets["particle_valid"].to(torch.float32))

        for hit, hit_cost_terms in self.hit_cost_weights.items():
            if hit == "sihit":
                base_cost_logits = (
                    torch.cat(
                        [outputs["flow_vtxd_logit"], outputs["flow_trkr_logit"]],
                        dim=-1,
                    ).detach().to(torch.float32)
                )
                cost_targets = torch.cat(
                    [targets["particle_vtxd_valid"], targets["particle_trkr_valid"]],
                    dim=-1,
                ).to(torch.float32)
                input_pad_mask = torch.cat([targets["vtxd_valid"], targets["trkr_valid"]], dim=-1)
            else:
                base_cost_logits = outputs[f"flow_{hit}_logit"].detach().to(torch.float32)
                cost_targets = targets[f"particle_{hit}_valid"].to(torch.float32)
                input_pad_mask = targets[f"{hit}_valid"]

            for cost_name, cost_weight in hit_cost_terms.items():
                cost_logits = base_cost_logits
                if cost_name == "mask_dice":
                    cost_logits = cost_logits * self.mask_dice_cost_logit_scale

                costs[f"{hit}_{cost_name}"] = float(cost_weight) * cost_fns[cost_name](
                    cost_logits,
                    cost_targets,
                    input_pad_mask=input_pad_mask,
                )

        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}

        part_class_idx = targets["particle_class_idx"].long()
        flow_class_logit = outputs["flow_logit"]
        dtype = flow_class_logit.dtype

        # Object class loss
        losses["object_class_ce"] = 0.5 * F.cross_entropy(
            flow_class_logit.flatten(0, 1),
            part_class_idx.flatten(0, 1),
            reduction="none",
        ).mean()

        # Compute the mask loss over all queries, even for null quries
        if self.loss_object_mask == "all":
            object_mask = torch.full_like(targets["particle_valid"], False, dtype=torch.bool)
        # Compute the mask loss only for queries matched to a valid particle
        elif self.loss_object_mask == "valid":
            object_mask = targets["particle_valid"]
        # Compute the mask loss only for queries matched to a particle that involves this hit type
        elif self.loss_object_mask == "selective":
            object_mask = torch.full_like(targets["particle_valid"], False, dtype=torch.bool)
            for class_name in self.class_name_to_idx:
                if class_name == "null":
                    continue
                object_mask = torch.logical_or(object_mask, targets[f"particle_is_{class_name}"])

        for hit, hit_loss_terms in self.hit_loss_weights.items():
            if hit == "sihit":
                loss_logits = torch.cat([outputs["flow_vtxd_logit"], outputs["flow_trkr_logit"]], dim=-1)
                loss_targets = torch.cat(
                    [targets["particle_vtxd_valid"], targets["particle_trkr_valid"]],
                    dim=-1,
                ).to(dtype)
                input_pad_mask = torch.cat([targets["vtxd_valid"], targets["trkr_valid"]], dim=-1)
            else:
                loss_logits = outputs[f"flow_{hit}_logit"]
                loss_targets = targets[f"particle_{hit}_valid"].to(dtype)
                input_pad_mask = targets[f"{hit}_valid"]

            for loss_name, loss_weight in hit_loss_terms.items():
                losses[f"{hit}_{loss_name}"] = float(loss_weight) * loss_fns[loss_name](
                    loss_logits,
                    loss_targets,
                    object_valid_mask=object_mask,
                    input_pad_mask=input_pad_mask,
                )

        if self.tracker_helix_fit:
            flow_helix_params = outputs["flow_regr_helix"]
            part_helix_params = torch.stack(
                [targets[f"particle_{field}"] for field in self.helix_fit_fields],
                dim=-1,
                )

            helix_fit_mask = (
                outputs["flow_helix_fitted"].bool()
                & targets["particle_is_primary"].bool()
            )

            if helix_fit_mask.any():
                losses["helix_l1"] = 0.001 * F.smooth_l1_loss(
                    flow_helix_params[helix_fit_mask],
                    part_helix_params[helix_fit_mask],
                ).mean()

        return losses

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {}
        eps = 1e-8

        has_sihits = {"vtxd", "trkr"}.issubset(set(self.hits_included))
        if has_sihits:
            preds["flow_sihits_valid"] = torch.cat([preds["flow_vtxd_valid"], preds["flow_trkr_valid"]], dim=-1)
            targets["particle_sihits_valid"] = torch.cat(
                [targets["particle_vtxd_valid"], targets["particle_trkr_valid"]], dim=-1
            )

        if self.tracker_helix_fit:
            helix_fit_mask = (
                preds["flow_helix_fitted"].bool()
                & targets["particle_is_primary"].bool()
            )
            helix_pred = torch.stack(
                [preds[f"flow_helix_{f}"] for f in self.helix_fit_fields],
                dim=-1,
            )
            helix_true = torch.stack(
                [targets[f"particle_{f}"] for f in self.helix_fit_fields],
                dim=-1,
            )

            if helix_fit_mask.any():
                diff = helix_pred[helix_fit_mask] - helix_true[helix_fit_mask]
                abs_err = diff.abs()
                rmse = torch.sqrt((diff ** 2).mean(dim=0))
                mae = abs_err.mean(dim=0)
                relerr = (abs_err / helix_true[helix_fit_mask].abs().clamp_min(eps)).mean(dim=0)
                ape_pct = relerr * 100.0
            else:
                zeros = preds["flow_logit"].new_zeros(len(self.helix_fit_fields))
                rmse = zeros
                mae = zeros
                relerr = zeros
                ape_pct = zeros

            for idx, field in enumerate(self.helix_fit_fields):
                field_key = field.replace(".", "_")
                metrics[f"helix_fit_rmse_{field_key}"] = rmse[idx]
                metrics[f"helix_fit_mae_{field_key}"] = mae[idx]
                metrics[f"helix_fit_relerr_{field_key}"] = relerr[idx]
                metrics[f"helix_fit_ape_pct_{field_key}"] = ape_pct[idx]

            charged_mask = targets["particle_is_charged"].bool()
            charged_total = charged_mask.float().sum()
            fittable = preds["flow_helix_fittable"].bool()
            fitted = preds["flow_helix_fitted"].bool()
            metrics["charged_helix_fittable_frac"] = (fittable & charged_mask).float().sum() / (charged_total + eps)
            metrics["charged_helix_fitted_frac"] = (fitted & charged_mask).float().sum() / (charged_total + eps)

        for selection in ["charged", "neutral", "electron", "charged_hadron", "neutral_hadron", "photon", "muon"]:
            part_selected = targets[f"particle_is_{selection}"].bool()
            flow_selected = preds[f"flow_is_{selection}"].bool()

            metrics[f"event_num_part_{selection}"] = part_selected.float().sum(-1).mean()
            metrics[f"event_num_flow_{selection}"] = flow_selected.float().sum(-1).mean()

            if self.tracker_helix_fit:
                metrics["event_num_flow_helix_fittable"] = preds["flow_helix_fittable"].float().sum(-1).mean()
                metrics["event_num_flow_helix_fitted"] = preds["flow_helix_fitted"].float().sum(-1).mean()

            active_hits = [h for h in self.class_active_hits[selection] if h in self.hits_included]
            if has_sihits and ("vtxd" in active_hits) and ("trkr" in active_hits):
                active_hits.append("sihits")

            for hit in active_hits:
                part_hit_valid = targets[f"particle_{hit}_valid"].bool()
                flow_hit_valid = preds[f"flow_{hit}_valid"].bool()

                part_hit_valid = part_hit_valid & part_selected.unsqueeze(-1)
                flow_hit_valid = flow_hit_valid & flow_selected.unsqueeze(-1)

                part_num_hit = part_hit_valid.float().sum(-1)
                flow_num_hit = flow_hit_valid.float().sum(-1)
                both_num_hit = (part_hit_valid & flow_hit_valid).float().sum(-1)

                metrics[f"part_{selection}_num_{hit}"] = part_num_hit[part_selected].mean()
                metrics[f"flow_{selection}_num_{hit}"] = flow_num_hit[flow_selected].mean()

                for wp in [0.5, 0.75, 1.0]:
                    part_is_eff = (both_num_hit / (part_num_hit + eps)) >= wp
                    flow_is_pur = (both_num_hit / (flow_num_hit + eps)) >= wp

                    eff = part_is_eff.float().sum() / (part_selected.float().sum() + eps)
                    pur = flow_is_pur.float().sum() / (flow_selected.float().sum() + eps)

                    metrics[f"{selection}_{hit}_eff_{wp}"] = eff
                    metrics[f"{selection}_{hit}_pur_{wp}"] = pur

        return metrics
