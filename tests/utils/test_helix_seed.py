from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from hepattn.utils.helix_fit import helix_seed_3pt

PLOT_DIR = Path(__file__).parent / "plots"


def _make_helix_hits(
    xc: float,
    yc: float,
    r: float,
    tan_lambda: float,
    z0: float,
    n_hits: int,
    phi_start: float,
    phi_end: float,
    ccw: bool = True,
):
    """Generate hits on a known helix. Returns x, y, z tensors of shape (n_hits,)."""
    phis = torch.linspace(phi_start, phi_end, n_hits, dtype=torch.float64)
    if not ccw:
        phis = phis.flip(0)
    x = xc + r * torch.cos(phis)
    y = yc + r * torch.sin(phis)
    # Arc-length from first hit
    s = r * (phis - phis[0])
    z = z0 + tan_lambda * s
    return x, y, z


class TestKnownCircle:
    """Construct hits on a known circle and verify recovered parameters."""

    @pytest.fixture
    def helix_params(self):
        return {"xc": 500.0, "yc": 300.0, "r": 400.0, "tan_lambda": 0.5, "z0": 10.0}

    @pytest.fixture
    def result(self, helix_params):
        x, y, z = _make_helix_hits(**helix_params, n_hits=9, phi_start=3.5, phi_end=4.5, ccw=True)
        valid = torch.ones(9, dtype=torch.bool)
        return helix_seed_3pt(x, y, z, valid, min_hits=5, eps=1e-12)

    def test_circle_center_and_radius(self, result, helix_params):
        assert result["xc"].item() == pytest.approx(helix_params["xc"], abs=1e-4)
        assert result["yc"].item() == pytest.approx(helix_params["yc"], abs=1e-4)
        assert result["R"].item() == pytest.approx(helix_params["r"], abs=1e-4)

    def test_tan_lambda(self, result, helix_params):
        assert result["tan_lambda"].item() == pytest.approx(helix_params["tan_lambda"], abs=1e-3)

    def test_output_keys(self, result):
        expected = {"d0", "phi0", "omega", "z0", "tan_lambda", "xc", "yc", "R", "theta", "eta"}
        assert set(result.keys()) == expected

    def test_plot_xy(self, result, helix_params):
        """Save xy-plane plot showing hits, fitted circle, selected hits, perigee."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        x, y, _z = _make_helix_hits(**helix_params, n_hits=9, phi_start=3.5, phi_end=4.5, ccw=True)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        # All hits
        ax.scatter(x.numpy(), y.numpy(), c="steelblue", s=40, zorder=3, label="Hits")

        # Highlight the 3 selected hits (first, middle, last by r_T)
        r_T = torch.sqrt(x**2 + y**2)
        order = torch.argsort(r_T)
        sel_idx = [order[0].item(), order[9 // 2].item(), order[8].item()]
        ax.scatter(
            x[sel_idx].numpy(),
            y[sel_idx].numpy(),
            c="red",
            s=120,
            marker="*",
            zorder=4,
            label="Selected 3",
        )

        # Fitted circle
        xc_fit = result["xc"].item()
        yc_fit = result["yc"].item()
        R_fit = result["R"].item()
        theta_plot = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            xc_fit + R_fit * np.cos(theta_plot),
            yc_fit + R_fit * np.sin(theta_plot),
            "k--",
            alpha=0.5,
            label="Fitted circle",
        )

        # Perigee
        rc = np.sqrt(xc_fit**2 + yc_fit**2)
        px = xc_fit - R_fit * xc_fit / rc
        py = yc_fit - R_fit * yc_fit / rc
        ax.scatter([px], [py], c="green", s=100, marker="D", zorder=5, label="Perigee")

        # IP
        ax.scatter([0], [0], c="black", s=80, marker="+", zorder=5, label="IP")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Helix seed 3pt — xy plane")
        ax.legend()
        ax.set_aspect("equal")
        fig.savefig(PLOT_DIR / "helix_seed_xy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def test_plot_yz(self, result, helix_params):
        """Save yz-plane plot."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        _x, y, z = _make_helix_hits(**helix_params, n_hits=9, phi_start=3.5, phi_end=4.5, ccw=True)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.scatter(y.numpy(), z.numpy(), c="steelblue", s=40, zorder=3, label="Hits")
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_title("Helix seed 3pt — yz plane")
        ax.legend()
        fig.savefig(PLOT_DIR / "helix_seed_yz.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


class TestChargeSign:
    """CW and CCW tracks should have opposite omega sign."""

    def test_omega_sign_flips(self):
        # Place the circle centre far from IP so r_T ordering is well-defined.
        # CCW track: hits go from phi=0.3 to phi=1.5 (increasing angle)
        xc, yc, radius = 800.0, 600.0, 200.0
        valid = torch.ones(9, dtype=torch.bool)

        x_ccw, y_ccw, z_ccw = _make_helix_hits(
            xc=xc,
            yc=yc,
            r=radius,
            tan_lambda=0.3,
            z0=0.0,
            n_hits=9,
            phi_start=0.3,
            phi_end=1.5,
            ccw=True,
        )
        res_ccw = helix_seed_3pt(x_ccw, y_ccw, z_ccw, valid, min_hits=5, eps=1e-12)

        # CW track: mirror the hits across the x-axis (negate y) to flip bending
        res_cw = helix_seed_3pt(x_ccw, -y_ccw, z_ccw, valid, min_hits=5, eps=1e-12)

        assert res_ccw["omega"].sign().item() != res_cw["omega"].sign().item()


class TestStraightTrack:
    """Collinear hits (R -> inf) should give omega ~ 0 and not crash."""

    def test_collinear_no_crash(self):
        n = 10
        x = torch.linspace(100.0, 1000.0, n, dtype=torch.float64)
        y = torch.linspace(50.0, 500.0, n, dtype=torch.float64)
        z = torch.linspace(0.0, 200.0, n, dtype=torch.float64)
        valid = torch.ones(n, dtype=torch.bool)

        result = helix_seed_3pt(x, y, z, valid, min_hits=5)
        # Very large R → very small omega
        assert result["omega"].abs().item() < 0.01


class TestMinHitsThreshold:
    """Track with fewer than min_hits valid hits should return zeros."""

    def test_too_few_hits(self):
        x = torch.randn(10, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)
        z = torch.randn(10, dtype=torch.float64)
        valid = torch.zeros(10, dtype=torch.bool)
        valid[:4] = True  # only 4 valid hits, min_hits=5

        result = helix_seed_3pt(x, y, z, valid, min_hits=5)
        for key in ("d0", "phi0", "omega", "z0", "tan_lambda", "xc", "yc", "R", "theta", "eta"):
            assert result[key].item() == 0.0


class TestBatched:
    """Multiple tracks in a batch with different valid counts."""

    def test_batch_shapes_and_correctness(self):
        B = 4
        M = 12
        x = torch.randn(B, M, dtype=torch.float64) * 500
        y = torch.randn(B, M, dtype=torch.float64) * 500
        z = torch.randn(B, M, dtype=torch.float64) * 200
        valid = torch.ones(B, M, dtype=torch.bool)
        # Make track 0 have only 3 valid hits (below threshold)
        valid[0, 3:] = False

        result = helix_seed_3pt(x, y, z, valid, min_hits=5)

        for key in result:
            assert result[key].shape == (B,), f"Shape mismatch for {key}"

        # Track 0 should be zeroed
        for key in result:
            assert result[key][0].item() == 0.0, f"Track 0 should be zero for {key}"

        # Tracks 1-3 should have non-zero R
        for i in range(1, B):
            assert result["R"][i].item() > 0.0


class TestNearLoopingArcLength:
    """Near-looping tracks (total arc ~ π) must not corrupt z regression."""

    def _run(self, ccw: bool):
        # CW helix: xc=-670, yc=-110, R=680, total arc ~2.9 rad (~0.92π)
        xc, yc, radius = -670.0, -110.0, 680.0
        tan_lambda, z0 = -0.98, 0.0
        # Arc span: 2.9 rad, well under π per consecutive pair of 9 hits
        phi_start, phi_end = 1.0, 1.0 + 2.9
        x, y, z = _make_helix_hits(
            xc=xc,
            yc=yc,
            r=radius,
            tan_lambda=tan_lambda,
            z0=z0,
            n_hits=9,
            phi_start=phi_start,
            phi_end=phi_end,
            ccw=ccw,
        )
        valid = torch.ones(9, dtype=torch.bool)
        result = helix_seed_3pt(x, y, z, valid, min_hits=5, eps=1e-12)
        assert result["tan_lambda"].item() == pytest.approx(tan_lambda if ccw else -tan_lambda, abs=0.05)
        assert result["z0"].item() == pytest.approx(z0, abs=5.0)

    def test_cw(self):
        self._run(ccw=False)

    def test_ccw(self):
        self._run(ccw=True)


class TestZFit:
    """Construct a helix with known tan_lambda and verify recovery."""

    @pytest.fixture
    def params(self):
        return {"xc": 400.0, "yc": -200.0, "r": 350.0, "tan_lambda": 1.2, "z0": -50.0}

    def test_tan_lambda_recovery(self, params):
        x, y, z = _make_helix_hits(
            **params,
            n_hits=11,
            phi_start=2.0,
            phi_end=3.0,
            ccw=True,
        )
        valid = torch.ones(11, dtype=torch.bool)
        result = helix_seed_3pt(x, y, z, valid, min_hits=5, eps=1e-12)
        assert result["tan_lambda"].item() == pytest.approx(params["tan_lambda"], abs=1e-2)

    def test_plot_z_vs_arclength(self, params):
        """Save z vs arc-length plot."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        x, y, z = _make_helix_hits(
            **params,
            n_hits=11,
            phi_start=2.0,
            phi_end=3.0,
            ccw=True,
        )
        valid = torch.ones(11, dtype=torch.bool)
        result = helix_seed_3pt(x, y, z, valid, min_hits=5, eps=1e-12)

        # Recompute arc-lengths for the 3 selected hits
        r_T = torch.sqrt(x**2 + y**2)
        order = torch.argsort(r_T)
        n = 11
        sel = [order[0].item(), order[n // 2].item(), order[n - 1].item()]

        xc_f, yc_f, R_f = result["xc"].item(), result["yc"].item(), result["R"].item()
        rc = np.sqrt(xc_f**2 + yc_f**2)
        px = xc_f - R_f * xc_f / rc
        py = yc_f - R_f * yc_f / rc
        alpha0 = np.arctan2(py - yc_f, px - xc_f)

        s_sel = []
        z_sel = []
        for i in sel:
            ai = np.arctan2(y[i].item() - yc_f, x[i].item() - xc_f)
            da = (ai - alpha0 + np.pi) % (2 * np.pi) - np.pi
            s_sel.append(R_f * da)
            z_sel.append(z[i].item())

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.scatter(s_sel, z_sel, c="red", s=80, zorder=3, label="Selected 3 hits")

        # Fitted line
        s_range = np.linspace(min(s_sel) - 20, max(s_sel) + 20, 100)
        z_line = result["z0"].item() + result["tan_lambda"].item() * s_range
        ax.plot(s_range, z_line, "k--", label=f"Fit: tan_lambda={result['tan_lambda'].item():.3f}")

        ax.set_xlabel("Arc-length s")
        ax.set_ylabel("z")
        ax.set_title("Helix seed 3pt — z vs arc-length")
        ax.legend()
        fig.savefig(PLOT_DIR / "helix_seed_z_vs_s.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
