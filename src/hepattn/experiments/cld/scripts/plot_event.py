from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.event_display import plot_cld_event

import yaml
import matplotlib.pyplot as plt


config_path = Path(__file__).parent.parent / "configs" / "combined_unified.yaml"
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0
config["test_dir"] = "/share/lustre/maxhart/data/cld/test_fix_prepped/reco_p8_ee_Zuds_ecm91_1"

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

for sample_id in sample_ids:
    sample = test_dataloader.dataset.load_sample(sample_id)
    filename = Path(test_dataloader.dataset.event_ids_to_event_filenames[sample_id]).stem
    inputs, targets = test_dataloader.dataset.prep_sample(sample)
    data = inputs | targets

    title_base = r"CLD Event " + str(sample_id) + r" (" + filename.replace("_", r"\_") + r")"

    pdf_path = out_dir / f"cld_event_{sample_id}.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plot_cld_event(data, axes_spec, "particle", usetex=True)
        try:
            fig.suptitle(title_base + r" — Truth")
            pdf.savefig(fig)
        finally:
            plt.close(fig)

        fig_pan = plot_cld_event(data, axes_spec, "pandora", usetex=True)
        try:
            fig_pan.suptitle(title_base + r" — Pandora")
            pdf.savefig(fig_pan)
        finally:
            plt.close(fig_pan)

    print(f"Saved event {sample_id} to {pdf_path}")
