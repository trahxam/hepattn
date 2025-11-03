import torch

from hepattn.experiments.cld.data import CLDDataset, CLDDataModule


class CLDParticleDataset(CLDDataset):
    def collate_fn(self, batch):
        inputs, targets = super().collate_fn(batch)

        particle_inputs, particle_targets = {}, {}
        valid_particles = targets["particle_valid"].bool()  # (B, N)
        B, N = valid_particles.shape

        # Flattened mask of which particles are valid across batch
        particle_valid = valid_particles.flatten()  # (B*N,)
        particle_valid = particle_valid[particle_valid]     # (Nv,)

        # --- Particle-level targets ---
        for field in self.targets["particle"]:
            particle_targets[f"particle_{field}"] = targets[f"particle_{field}"][valid_particles]

        # --- Per-hit particle inputs ---
        for hit, fields in self.inputs.items():
            if hit == "sihit":
                continue

            mask = targets[f"particle_{hit}_valid"]  # (B, N, M)
            M = mask.shape[-1]

            mask_v = mask[valid_particles]  # (Nv, M)

            sorted_idx = torch.argsort(mask_v, dim=1, descending=True)  # (Nv, M)
            hit_counts = mask_v.sum(dim=1)
            max_hits = int(hit_counts.max().item())

            hit_idx = sorted_idx[:, :max_hits]
            particle_hit_valid = (
                torch.arange(max_hits, device=mask.device)[None, :] < hit_counts[:, None]
            )  # (Nv, max_hits)

            per_batch_valid = targets["particle_valid"].sum(dim=1)
            batch_idx = torch.arange(B, device=mask.device).repeat_interleave(per_batch_valid)
            offsets = (batch_idx * M).unsqueeze(1)

            gather_idx = hit_idx + offsets

            for field in fields:
                feats = inputs[f"{hit}_{field}"]
                feats_flat = feats.reshape(B * M, -1) if feats.ndim == 3 else feats.reshape(B * M)
                gathered = feats_flat[gather_idx]
                particle_inputs[f"particle_{hit}_{field}"] = gathered

            particle_inputs[f"particle_{hit}_valid"] = particle_hit_valid

        # Also include which particles are valid
        particle_targets["particle_valid"] = particle_valid

        return particle_inputs, particle_targets


class CLDParticleDataModule(CLDDataModule):
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dset = CLDParticleDataset(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs)
            self.val_dset = CLDParticleDataset(dirpath=self.val_dir, num_samples=self.num_val, **self.kwargs)
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = CLDParticleDataset(dirpath=self.test_dir, num_samples=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")
