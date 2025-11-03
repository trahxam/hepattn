from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# If you saved CLDParticleDataset next to CLDDataset:
from hepattn.experiments.cld.fitting.data import CLDDataModule, CLDParticleDataset


# Minimal DataModule that swaps in CLDParticleDataset
class CLDParticleDataModule(CLDDataModule):
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dset = CLDParticleDataset(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs)
            self.val_dset   = CLDParticleDataset(dirpath=self.val_dir,   num_samples=self.num_val,   **self.kwargs)
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = CLDParticleDataset(dirpath=self.test_dir, num_samples=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")


# ---- load config like your example ----
config_path = Path("src/hepattn/experiments/cld/fitting/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0  # like your snippet

# ---- build particle-level datamodule & dataloader ----
datamodule = CLDParticleDataModule(**config)
datamodule.setup(stage="test")
test_dataloader = datamodule.test_dataloader()
data_iter = iter(test_dataloader)

# ---- read one batch (particle-level) ----


inputs, targets = next(data_iter)

for k, v in inputs.items():
    print("input:  ", k, v.shape)

for k, v in targets.items():
    print("target: ", k, v.shape)


plt.rcParams["figure.dpi"] = 300


fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

for i in range(16):
    for hit in ["vtxd", "trkr"]:
        valid = inputs[f"particle_{hit}_valid"][i]

        hit_x = inputs[f"particle_{hit}_pos.x"][i][valid]
        hit_y = inputs[f"particle_{hit}_pos.y"][i][valid]

        # Have to convert vertex from mm to m
        particle_vx = 1e-3 * targets["particle_vtx.x"][i]
        particle_vy = 1e-3 * targets["particle_vtx.y"][i]

        particle_px = targets["particle_mom.x"][i]
        particle_py = targets["particle_mom.y"][i]
        particle_pt = targets["particle_mom.r"][i]

        hit_color = {"vtxd": "purple", "trkr": "blue"}[hit]

        axes[i].scatter(hit_x, hit_y, s=8, color=hit_color, marker="x")
        axes[i].scatter(particle_vx, particle_vy, s=64, color="black", marker="+")

        pt_scale = 0.25

        axes[i].annotate(
            "",
            xy=(particle_vx + pt_scale * particle_px / particle_pt,
                particle_vy + pt_scale * particle_py / particle_pt),
            xytext=(particle_vx, particle_vy),
            arrowprops=dict(arrowstyle="->", linewidth=0.5, color="black"),
        )

fig.tight_layout()
fig.savefig("src/hepattn/experiments/cld/plots/particle_displays.png")
plt.close()