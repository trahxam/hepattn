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
        sihit_gated_calo_cost: bool = False,
        class_conditional_cost: bool = True,
        particle_cost_hits: dict[str, list[str]] | None = None,
        per_hit_class_loss_mask: bool = False,
        return_embeddings: bool = False,
        task_stage: int = 0,
        charged_query_slots: list[int] | None = None,
        neutral_query_slots: list[int] | None = None,
        class_query_slots: dict[str, list[int]] | None = None,
        use_slot_class: bool = False,
        class_valid_hit: dict[str, str] | None = None,
        mask_unphysical_hits: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss, task_stage=task_stage)

        self.name = name
        self.dim = dim
        self.mask_attn = mask_attn
        self.loss_object_mask = loss_object_mask
        self.calo_score_method = calo_score_method
        self.return_embeddings = return_embeddings
        self.hit_loss_weights = hit_loss_weights or {}
        self.hit_cost_weights = hit_cost_weights or {}
        self.mask_dice_cost_logit_scale = float(mask_dice_cost_logit_scale)
        self.sihit_gated_calo_cost = sihit_gated_calo_cost
        self.class_conditional_cost = class_conditional_cost
        self.per_hit_class_loss_mask = per_hit_class_loss_mask

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

        # Precompute: for each hit (including "sihit"), which class indices should activate it.
        # Only real classes (in class_name_to_idx) are included; meta-classes like "charged"/"neutral" are skipped.
        self.hit_active_class_idxs: dict[str, list[int]] = {}
        for hit, classes in self.hit_active_classes.items():
            self.hit_active_class_idxs[hit] = [
                self.class_name_to_idx[c] for c in classes if c in self.class_name_to_idx
            ]
        # sihit is the union of vtxd and trkr active classes
        sihit_classes = set(self.hit_active_classes.get("vtxd", [])) | set(self.hit_active_classes.get("trkr", []))
        self.hit_active_class_idxs["sihit"] = [
            self.class_name_to_idx[c] for c in sihit_classes if c in self.class_name_to_idx
        ]

        # Optional per-particle override for the cost (independent of the loss masking).
        # Specified as particle_class -> [hit_types]; inverted to hit -> [class_idxs] for the cost loop.
        # If not provided, falls back to hit_active_class_idxs (i.e. all hits expected for each class).
        if particle_cost_hits is not None:
            self.hit_cost_class_idxs: dict[str, list[int]] = {}
            for class_name, hits in particle_cost_hits.items():
                if class_name not in self.class_name_to_idx:
                    continue
                class_idx = self.class_name_to_idx[class_name]
                for hit in hits:
                    self.hit_cost_class_idxs.setdefault(hit, []).append(class_idx)
        else:
            self.hit_cost_class_idxs = self.hit_active_class_idxs

        self.charged_classes = ["charged_hadron", "electron", "muon"]
        self.neutral_classes = ["neutral_hadron", "photon"]

        self.register_buffer("charged_class_idxs", torch.tensor([2, 4, 5], dtype=torch.long))
        self.register_buffer("neutral_class_idxs", torch.tensor([1, 3], dtype=torch.long))

        # Partition query slots: restrict which queries can be matched to charged vs neutral particles
        self.charged_query_slots = charged_query_slots  # [start, end] inclusive-exclusive
        self.neutral_query_slots = neutral_query_slots  # [start, end] inclusive-exclusive
        # Fine-grained per-class slot partitioning: each range is reserved for one class
        self.class_query_slots = class_query_slots  # {class_name: [start, end]}
        # When True: class is determined by slot assignment; class_net only predicts valid/null
        self.use_slot_class = use_slot_class and class_query_slots is not None
        self.class_valid_hit = class_valid_hit  # maps class_name -> hit type for per-slot valid/null prediction
        self.mask_unphysical_hits = mask_unphysical_hits and class_query_slots is not None

        # Network for particle classification (binary valid/null when slot class is used)
        self.class_net = Dense(dim, 1 if self.use_slot_class else len(self.class_name_to_idx))

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

        if self.use_slot_class:
            B, N_q, _ = x["query_embed"].shape
            if self.class_valid_hit is not None and self.class_query_slots is not None:
                # Use per-type query embeddings for each slot range to avoid cross-type mixing.
                # Each class's slot range uses only its associated hit type's query embedding.
                valid_logit = x["query_embed"].new_empty(B, N_q)
                for class_name, (s, e) in self.class_query_slots.items():
                    hit = self.class_valid_hit[class_name]
                    q = x.get(f"query_embed_{hit}", x["query_embed"])
                    valid_logit[:, s:e] = self.class_net(q[:, s:e]).squeeze(-1)
            else:
                valid_logit = self.class_net(x["query_embed"]).squeeze(-1)  # [B, N_q]

            # Build slot→class index mapping (cached after first call)
            if not hasattr(self, "_slot_class_idx") or self._slot_class_idx.shape[0] != N_q:
                slot_class = valid_logit.new_zeros(N_q, dtype=torch.long)
                for class_name, (s, e) in self.class_query_slots.items():
                    slot_class[s:e] = self.class_name_to_idx[class_name]
                self._slot_class_idx = slot_class

            # Synthetic 6-class logit: null=0 (reference), assigned class=valid_logit, others=-1e4
            flow_logit = valid_logit.new_full((B, N_q, len(self.class_name_to_idx)), -1e4)
            flow_logit[:, :, 0] = 0.0
            flow_logit.scatter_(2, self._slot_class_idx[None, :, None].expand(B, -1, 1), valid_logit[:, :, None])
            outputs["flow_logit"] = flow_logit
        else:
            outputs["flow_logit"] = self.class_net(x["query_embed"])

        if self.return_embeddings:
            outputs["query_embed"] = x["query_embed"]

        for hit, mask_net in self.hit_mask_nets.items():
            # query-side mask embedding: use per-type query if available, else shared
            q = mask_net(x.get(f"query_embed_{hit}", x["query_embed"]))  # [B, Nq, C]
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

            # Set logits to -inf for slot ranges where this hit type is physically inactive
            if self.mask_unphysical_hits:
                for class_name, (s, e) in self.class_query_slots.items():
                    if hit not in self.class_active_hits.get(class_name, []):
                        flow_hit_logit[:, s:e, :] = torch.finfo(flow_hit_logit.dtype).min

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

        # Partition cost: large penalty when charged slots match neutral particles or vice versa
        if self.charged_query_slots is not None or self.neutral_query_slots is not None:
            B, N_q = flow_class_logit.shape[:2]
            N_p = targets["particle_class_idx"].shape[1]
            part_class = targets["particle_class_idx"]  # (B, N_p)
            is_charged = torch.isin(part_class, self.charged_class_idxs)  # (B, N_p)
            is_neutral = torch.isin(part_class, self.neutral_class_idxs)  # (B, N_p)
            penalty = flow_class_logit.new_zeros(B, N_q, N_p)
            if self.charged_query_slots is not None:
                cs, ce = self.charged_query_slots
                # Charged slots should not be matched to neutral particles
                penalty[:, cs:ce, :] += 1e4 * is_neutral.unsqueeze(1).float()
            if self.neutral_query_slots is not None:
                ns, ne = self.neutral_query_slots
                # Neutral slots should not be matched to charged particles
                penalty[:, ns:ne, :] += 1e4 * is_charged.unsqueeze(1).float()
            costs["partition"] = penalty

        # Fine-grained per-class slot partitioning
        if self.class_query_slots is not None:
            B, N_q = flow_class_logit.shape[:2]
            N_p = targets["particle_class_idx"].shape[1]
            part_class = targets["particle_class_idx"]  # (B, N_p)
            part_valid = targets["particle_valid"]       # (B, N_p)
            penalty = flow_class_logit.new_zeros(B, N_q, N_p)
            for class_name, (s, e) in self.class_query_slots.items():
                target_idx = self.class_name_to_idx[class_name]
                # Penalise slots reserved for class_name when the target particle is a
                # different (valid) class; null padding targets are never penalised.
                wrong_class = part_valid & (part_class != target_idx)
                penalty[:, s:e, :] += 1e4 * wrong_class.unsqueeze(1).float()
            costs["partition"] = costs.get("partition", flow_class_logit.new_zeros(B, N_q, N_p)) + penalty

        # Pre-compute sihit cost for gating calo costs (only if needed)
        sihit_gate = None
        if self.sihit_gated_calo_cost and "sihit" in self.hit_cost_weights:
            sihit_logits = torch.cat(
                [outputs["flow_vtxd_logit"], outputs["flow_trkr_logit"]], dim=-1
            ).detach().to(torch.float32)
            sihit_targets = torch.cat(
                [targets["particle_vtxd_valid"], targets["particle_trkr_valid"]], dim=-1
            ).to(torch.float32)
            sihit_pad_mask = torch.cat([targets["vtxd_valid"], targets["trkr_valid"]], dim=-1)
            sihit_gate = cost_fns["mask_dice"](sihit_logits, sihit_targets, input_pad_mask=sihit_pad_mask)

        # Hits that are considered tracker-type (not gated by sihit)
        tracker_hits = {"sihit", "vtxd", "trkr"}

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

                c = float(cost_weight) * cost_fns[cost_name](
                    cost_logits,
                    cost_targets,
                    input_pad_mask=input_pad_mask,
                )

                # Gate calo costs by sihit cost: calo only contributes when sihit is uncertain
                if sihit_gate is not None and hit not in tracker_hits:
                    c = c * sihit_gate

                # Zero out cost for particles whose class should not activate this hit type
                if self.class_conditional_cost:
                    active_idxs = self.hit_cost_class_idxs.get(hit, [])
                    part_class = targets["particle_class_idx"]  # (B, N_particles)
                    class_mask = torch.zeros_like(part_class, dtype=c.dtype)
                    for idx in active_idxs:
                        class_mask = class_mask + (part_class == idx).to(c.dtype)
                    c = c * class_mask.unsqueeze(1)  # (B, 1, N_particles)

                costs[f"{hit}_{cost_name}"] = c

        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}

        part_class_idx = targets["particle_class_idx"].long()
        flow_class_logit = outputs["flow_logit"]
        dtype = flow_class_logit.dtype

        # Object class loss
        if self.use_slot_class:
            # Class is determined by slot; only predict valid/null.
            # valid_logit = class_net output; in the synthetic flow_logit: null=0, assigned=valid_logit
            # So valid_logit = flow_logit[assigned_class] - flow_logit[null=0]
            slot_cls = self._slot_class_idx if hasattr(self, "_slot_class_idx") else None
            if slot_cls is not None:
                assigned_logit = flow_class_logit.gather(2, slot_cls[None, :, None].expand(flow_class_logit.shape[0], -1, 1)).squeeze(-1)
            else:
                assigned_logit = flow_class_logit[..., 1:].max(-1).values
            losses["object_valid_bce"] = 0.5 * F.binary_cross_entropy_with_logits(
                assigned_logit,
                targets["particle_valid"].to(dtype),
                reduction="mean",
            )
        else:
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
                active_idxs = self.hit_active_class_idxs.get("sihit", [])
            else:
                loss_logits = outputs[f"flow_{hit}_logit"]
                loss_targets = targets[f"particle_{hit}_valid"].to(dtype)
                input_pad_mask = targets[f"{hit}_valid"]
                active_idxs = self.hit_active_class_idxs.get(hit, [])

            # Optionally restrict loss to particles whose class activates this hit type,
            # preventing e.g. photon queries from being trained to suppress silicon hits.
            if self.per_hit_class_loss_mask and active_idxs:
                hit_object_mask = torch.zeros_like(targets["particle_valid"], dtype=torch.bool)
                for idx in active_idxs:
                    hit_object_mask |= (targets["particle_class_idx"] == idx)
                effective_mask = object_mask & hit_object_mask
            else:
                effective_mask = object_mask

            for loss_name, loss_weight in hit_loss_terms.items():
                losses[f"{hit}_{loss_name}"] = float(loss_weight) * loss_fns[loss_name](
                    loss_logits,
                    loss_targets,
                    object_valid_mask=effective_mask,
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
