"""CLD track-fitting dataset.

Converts per-event CLD data (with Pandora PFO hit-assignment dense masks) into
per-track tensors suitable for BoostedTrackFitter.  Each charged Pandora PFO
becomes one "track", with vtxd + trkr silicon hits gathered from the dense
pandora_{vtxd,trkr}_valid masks and matched to the best truth particle via IoU.

Supports batch_size > 1: events are processed independently and their tracks
concatenated into a single flat track list.  Because BoostedTrackFitter has no
inter-track attention, tracks from different events are fully independent and
can be batched without any masking.  The global sihit array layout is:

    event 0 hits | event 1 hits | ...   (each block = N_vtxd_padded + N_trkr_padded)

CSR indices are offset accordingly so that every track points into the correct
slice of the global sihit array.
"""

import torch
from torch import Tensor

from hepattn.experiments.cld.data import CLDDataModule, CLDDataset


def _dense_mask_to_csr(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Convert a dense boolean mask (N_tracks, N_hits) to CSR (indptr, indices).

    Returns:
        indptr:  (N_tracks+1,) long tensor of row pointers.
        indices: (N_nonzero,) long tensor of column indices (hit indices per track).
    """
    counts = mask.long().sum(dim=1)                 # (N_tracks,)
    indptr = torch.zeros(mask.shape[0] + 1, dtype=torch.long, device=mask.device)
    indptr[1:] = counts.cumsum(0)
    indices = mask.nonzero(as_tuple=False)[:, 1]    # (N_nonzero,)
    return indptr, indices


class CLDTrackDataset(CLDDataset):
    """Wraps CLDDataset to produce per-track inputs for BoostedTrackFitter.

    Expected config fields (inputs / targets):
        inputs.vtxd:  [pos.x, pos.y, pos.z, pos.r, pos.s, pos.eta, pos.phi, ...]
        inputs.trkr:  [pos.x, pos.y, pos.z, pos.r, pos.s, pos.eta, pos.phi, ...]
        targets.pandora:      [mom.theta, mom.phi, mom.qopt, perigee.d0, perigee.z0, charge, is_charged]
        targets.pandora_vtxd: []  (only the valid mask is used)
        targets.pandora_trkr: []
        targets.particle:     [mom.theta, mom.phi, mom.qopt, perigee.d0, perigee.z0]
        targets.particle_vtxd: []
        targets.particle_trkr: []

    Returned keys (after collate_fn, N_tracks = sum of charged Pandora PFOs across all events):
        inputs:
            sihit_{x,y,z,r,s,eta,phi}:   (1, B*N_sihit_per_event)  merged hit coords [m]
            track_sihit_indptr:            (1, N_tracks+1)
            track_sihit_indices:           (1, N_assignments)
        targets:
            track_matched_particle_theta:        (1, N_tracks)
            track_matched_particle_phi_perigee:  (1, N_tracks)
            track_matched_particle_qopt:         (1, N_tracks)
            track_matched_particle_d0_perigee_m: (1, N_tracks)  [metres]
            track_matched_particle_z0_perigee_m: (1, N_tracks)  [metres]
            track_matched_particle_valid:        (1, N_tracks)  bool
            track_theta: (1, N_tracks)  Pandora theta (baseline)
            track_phi:   (1, N_tracks)  Pandora phi   (baseline)
            track_qopt:  (1, N_tracks)  Pandora q/pT  (baseline)
            track_d0:    (1, N_tracks)  Pandora d0 [m] (baseline)
            track_z0:    (1, N_tracks)  Pandora z0 [m] (baseline)
    """

    match_min_iou: float = 0.5

    def collate_fn(
        self,
        batch: list[tuple[dict[str, Tensor], dict[str, Tensor]]],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        inputs, targets = super().collate_fn(batch)

        B = next(iter(inputs.values())).shape[0]

        hit_fields  = ["pos.x", "pos.y", "pos.z", "pos.r", "pos.s", "pos.theta", "pos.phi"]
        sihit_short = ["x",     "y",     "z",     "r",     "s",     "theta",    "phi"]

        # Each event in the batch is padded to the same size.
        # N_sihit_per_event = N_vtxd_padded + N_trkr_padded (constant across batch).
        N_vtxd_padded = inputs[f"vtxd_{hit_fields[0]}"].shape[1]
        N_trkr_padded = inputs[f"trkr_{hit_fields[0]}"].shape[1]
        N_sihit_per_event = N_vtxd_padded + N_trkr_padded

        # ── Merge hit features across all events ─────────────────────────────
        # Combined sihit array: for event b its hits occupy
        #   [b*N_sihit_per_event : (b+1)*N_sihit_per_event]
        # Separate vtxd/trkr arrays: event b's vtxd hits are at
        #   [b*N_vtxd_padded : (b+1)*N_vtxd_padded]
        # and trkr hits at [b*N_trkr_padded : (b+1)*N_trkr_padded].
        track_inputs: dict[str, Tensor] = {}
        for field, short in zip(hit_fields, sihit_short):
            track_inputs[f"sihit_{short}"] = torch.cat(
                [inputs[f"vtxd_{field}"], inputs[f"trkr_{field}"]], dim=1
            ).reshape(1, -1)   # (1, B * N_sihit_per_event)
            track_inputs[f"vtxd_{short}"] = inputs[f"vtxd_{field}"].reshape(1, -1)  # (1, B*N_vtxd_padded)
            track_inputs[f"trkr_{short}"] = inputs[f"trkr_{field}"].reshape(1, -1)  # (1, B*N_trkr_padded)

        # Binary detector-type flag in combined sihit: 1.0 for vtxd, 0.0 for trkr.
        # Used by the helix fit for vtxd-only z(r) weighting.
        vtxd_flag = inputs[f"vtxd_{hit_fields[0]}"].new_ones( inputs[f"vtxd_{hit_fields[0]}"].shape)
        trkr_flag = inputs[f"trkr_{hit_fields[0]}"].new_zeros(inputs[f"trkr_{hit_fields[0]}"].shape)
        track_inputs["sihit_is_vtxd"] = torch.cat([vtxd_flag, trkr_flag], dim=1).reshape(1, -1)

        # ── Per-event track building ──────────────────────────────────────────
        indptr_parts: list[Tensor] = []
        indices_parts: list[Tensor] = []
        cumulative_assignments = 0

        vtxd_indptr_parts: list[Tensor] = []
        vtxd_indices_parts: list[Tensor] = []
        cumulative_vtxd = 0

        trkr_indptr_parts: list[Tensor] = []
        trkr_indices_parts: list[Tensor] = []
        cumulative_trkr = 0

        pan_theta_list: list[Tensor] = []
        pan_phi_list:   list[Tensor] = []
        pan_qopt_list:  list[Tensor] = []
        pan_d0_list:    list[Tensor] = []
        pan_z0_list:    list[Tensor] = []

        truth_theta_list: list[Tensor] = []
        truth_phi_list:   list[Tensor] = []
        truth_qopt_list:  list[Tensor] = []
        truth_d0_list:    list[Tensor] = []
        truth_z0_list:    list[Tensor] = []
        matched_list:     list[Tensor] = []

        for b in range(B):
            # ── Charged, valid Pandora PFOs ───────────────────────────────
            pandora_valid   = targets["pandora_valid"][b].bool()
            pandora_charged = targets["pandora_is_charged"][b].bool()
            charged_mask    = pandora_valid & pandora_charged
            charged_idx     = charged_mask.nonzero(as_tuple=True)[0]  # (N_charged,)

            # ── Pandora baseline params ───────────────────────────────────
            pan_theta_list.append(targets["pandora_mom.theta"][b][charged_idx])
            pan_phi_list.append( targets["pandora_mom.phi"][b][charged_idx])
            pan_qopt_list.append(targets["pandora_mom.qopt"][b][charged_idx])
            pan_d0_list.append(  targets["pandora_perigee.d0"][b][charged_idx] * 1e-3)  # mm → m
            pan_z0_list.append(  targets["pandora_perigee.z0"][b][charged_idx] * 1e-3)

            # ── Hit masks for charged Pandora PFOs ────────────────────────
            pan_vtxd  = targets["pandora_vtxd_valid"][b][charged_idx].bool()  # (N_charged, N_vtxd)
            pan_trkr  = targets["pandora_trkr_valid"][b][charged_idx].bool()  # (N_charged, N_trkr)
            pan_sihit = torch.cat([pan_vtxd, pan_trkr], dim=1)               # (N_charged, N_sihit_per_event)

            # ── CSR for this event, shifted into global sihit array ───────
            indptr_b, indices_b = _dense_mask_to_csr(pan_sihit)

            # Append indptr (skip leading 0 for all but the first event)
            if not indptr_parts:
                indptr_parts.append(indptr_b)
            else:
                indptr_parts.append(indptr_b[1:] + cumulative_assignments)

            # Shift hit indices into the global sihit array for this event
            indices_parts.append(indices_b + b * N_sihit_per_event)
            cumulative_assignments += int(indptr_b[-1].item())

            # ── Separate vtxd / trkr CSRs (for per-type embedding) ───────
            indptr_vtxd_b, indices_vtxd_b = _dense_mask_to_csr(pan_vtxd)
            if not vtxd_indptr_parts:
                vtxd_indptr_parts.append(indptr_vtxd_b)
            else:
                vtxd_indptr_parts.append(indptr_vtxd_b[1:] + cumulative_vtxd)
            vtxd_indices_parts.append(indices_vtxd_b + b * N_vtxd_padded)
            cumulative_vtxd += int(indptr_vtxd_b[-1].item())

            indptr_trkr_b, indices_trkr_b = _dense_mask_to_csr(pan_trkr)
            if not trkr_indptr_parts:
                trkr_indptr_parts.append(indptr_trkr_b)
            else:
                trkr_indptr_parts.append(indptr_trkr_b[1:] + cumulative_trkr)
            trkr_indices_parts.append(indices_trkr_b + b * N_trkr_padded)
            cumulative_trkr += int(indptr_trkr_b[-1].item())

            # ── Pandora → particle IoU matching ──────────────────────────
            par_valid = targets["particle_valid"][b].bool()
            par_vtxd  = targets["particle_vtxd_valid"][b].bool()
            par_trkr  = targets["particle_trkr_valid"][b].bool()
            par_sihit = torch.cat([par_vtxd, par_trkr], dim=1).float()  # (N_par, N_sihit_per_event)

            pan_sihit_f = pan_sihit.float()
            intersection = pan_sihit_f @ par_sihit.T                    # (N_charged, N_par)
            pan_nhits    = pan_sihit_f.sum(dim=1, keepdim=True)         # (N_charged, 1)
            par_nhits    = par_sihit.sum(dim=1, keepdim=True)           # (N_par, 1)
            union        = pan_nhits + par_nhits.T - intersection
            iou          = intersection / union.clamp(min=1.0)
            iou[:, ~par_valid] = 0.0

            best_iou, best_par_idx = iou.max(dim=1)
            matched = best_iou >= self.match_min_iou

            # ── Truth fields for matched particles ────────────────────────
            safe_idx   = best_par_idx.clamp(min=0)
            truth_theta = targets["particle_mom.theta"][b][safe_idx]
            truth_phi   = targets["particle_mom.phi"][b][safe_idx]
            truth_qopt  = targets["particle_mom.qopt"][b][safe_idx]
            truth_d0_m  = targets["particle_perigee.d0"][b][safe_idx] * 1e-3   # mm → m
            truth_z0_m  = targets["particle_perigee.z0"][b][safe_idx] * 1e-3

            truth_theta[~matched] = 0.0
            truth_phi[~matched]   = 0.0
            truth_qopt[~matched]  = 0.0
            truth_d0_m[~matched]  = 0.0
            truth_z0_m[~matched]  = 0.0

            truth_theta_list.append(truth_theta)
            truth_phi_list.append(truth_phi)
            truth_qopt_list.append(truth_qopt)
            truth_d0_list.append(truth_d0_m)
            truth_z0_list.append(truth_z0_m)
            matched_list.append(matched)

        # ── Combine CSR across events ─────────────────────────────────────────
        track_inputs["track_sihit_indptr"]  = torch.cat(indptr_parts).unsqueeze(0)
        track_inputs["track_sihit_indices"] = torch.cat(indices_parts).unsqueeze(0) if indices_parts else torch.zeros(1, 0, dtype=torch.long)

        track_inputs["track_vtxd_indptr"]  = torch.cat(vtxd_indptr_parts).unsqueeze(0)
        track_inputs["track_vtxd_indices"] = torch.cat(vtxd_indices_parts).unsqueeze(0) if vtxd_indices_parts else torch.zeros(1, 0, dtype=torch.long)

        track_inputs["track_trkr_indptr"]  = torch.cat(trkr_indptr_parts).unsqueeze(0)
        track_inputs["track_trkr_indices"] = torch.cat(trkr_indices_parts).unsqueeze(0) if trkr_indices_parts else torch.zeros(1, 0, dtype=torch.long)

        # ── Combine track-level targets across events ─────────────────────────
        track_targets: dict[str, Tensor] = {
            "track_matched_particle_theta":        torch.cat(truth_theta_list).unsqueeze(0),
            "track_matched_particle_phi_perigee":  torch.cat(truth_phi_list).unsqueeze(0),
            "track_matched_particle_qopt":         torch.cat(truth_qopt_list).unsqueeze(0),
            "track_matched_particle_d0_perigee_m": torch.cat(truth_d0_list).unsqueeze(0),
            "track_matched_particle_z0_perigee_m": torch.cat(truth_z0_list).unsqueeze(0),
            "track_matched_particle_valid":        torch.cat(matched_list).unsqueeze(0),
            "track_theta": torch.cat(pan_theta_list).unsqueeze(0),
            "track_phi":   torch.cat(pan_phi_list).unsqueeze(0),
            "track_qopt":  torch.cat(pan_qopt_list).unsqueeze(0),
            "track_d0":    torch.cat(pan_d0_list).unsqueeze(0),   # [m]
            "track_z0":    torch.cat(pan_z0_list).unsqueeze(0),   # [m]
        }

        return track_inputs, track_targets


class CLDTrackDataModule(CLDDataModule):
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dset = CLDTrackDataset(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs)
            self.val_dset   = CLDTrackDataset(dirpath=self.val_dir,   num_samples=self.num_val,   **self.kwargs)
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = CLDTrackDataset(dirpath=self.test_dir, num_samples=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")
