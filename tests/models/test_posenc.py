from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hepattn.models.posenc2 import pos_enc, pos_enc_symmetric


def test_pos_enc():
    xs = torch.linspace(-torch.pi, torch.pi, 1000)
    dim = 128
    out_dir = Path("tests/pe")
    out_dir.mkdir(exist_ok=True)
    for alpha in [10, 20, 50, 100]:
        pe = pos_enc(xs, dim, alpha)
        pe_sym = pos_enc_symmetric(xs, dim, alpha)

        # display the positional encoding itself
        plt.figure()
        plt.imshow(pe.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_{alpha}.png")
        plt.close()

        plt.figure()
        plt.imshow(pe_sym.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_sym_{alpha}.png")
        plt.close()

        sim = pe @ pe.T
        sim_sym = pe_sym @ pe_sym.T
        plt.figure()
        plt.imshow(sim.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_{alpha}.png")
        plt.close()

        plt.figure()
        plt.imshow(sim_sym.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_sym_{alpha}.png")
        plt.close()
