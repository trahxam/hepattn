from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.event_display import plot_cld_event

import yaml
import matplotlib.pyplot as plt


config_path = Path(__file__).parent.parent / "configs" / "combined_unified.yaml"
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0
config["test_dir"] = "/share/rcif2/maxhart/data/cld/prepped/temp/reco_p8_ee_Zuds_ecm91_11112463_6_condor/"

datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")

test_dataloader = datamodule.test_dataloader()

out_dir = Path(__file__).parent.parent / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

sample_ids = test_dataloader.dataset.sample_ids[:10]

axes_spec = [
    {
        "x": "pos.x",
        "y": "pos.y",
        "px": "mom.x",
        "py": "mom.y",
        "input_names": ["vtxd", "trkr", "ecal", "hcal", "muon"],
    },
    {
        "x": "pos.z",
        "y": "pos.y",
        "px": "mom.z",
        "py": "mom.y",
        "input_names": ["vtxd", "trkr", "ecal", "hcal", "muon"],
    },
]

pdf_path = out_dir / "cld_events.pdf"
with PdfPages(pdf_path) as pdf:
    for sample_id in sample_ids:
        sample = test_dataloader.dataset.load_sample(sample_id)
        filename = Path(test_dataloader.dataset.event_ids_to_event_filenames[sample_id]).stem
        inputs, targets = test_dataloader.dataset.prep_sample(sample)
        data = inputs | targets

        fig = plot_cld_event(data, axes_spec, "particle")
        fig.axes[0].set_xlim(-3.5, 3.5)
        fig.axes[0].set_ylim(-3.5, 3.5)
        fig.axes[1].set_xlim(-5.0, 5.0)
        fig.axes[1].set_ylim(-3.5, 3.5)
        fig.suptitle(f"CLD Event {sample_id} ({filename})")

        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved all events to {pdf_path}")
