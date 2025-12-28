import yaml
from pathlib import Path
from hepattn.experiments.top.data import TopEventDataModule
from hepattn.experiments.top.event_display import plot_top_events

cfg_path = Path("src/hepattn/experiments/top/configs/base.yaml")
out_dir = Path("src/hepattn/experiments/top/plots")
out_dir.mkdir(parents=True, exist_ok=True)

with cfg_path.open() as f:
    cfg = yaml.safe_load(f)

dm = TopEventDataModule(**dict(cfg["data"]))
dm.setup("fit")
loader = dm.train_dataloader()

# Read one batch, merge inputs + targets into one dict
inputs, targets = next(iter(loader))


fig = plot_top_events(
    inputs,
    targets,
    nplots=36,
    ncols=6,
    seed=123,
)

fig.savefig(out_dir / "event_displays.png")
