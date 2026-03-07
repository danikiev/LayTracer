r"""
Quality test: two-layer model with identical parameters must match homogeneous solution.

Compares three approaches for every computed quantity
(travel time, t*, geometrical spreading, transmission coefficient, and their product):

    a) Analytical homogeneous solution (closed-form formulas)
    b) Homogeneous solution using LayTracer (single-layer model)
    c) Layered solution using LayTracer (two layers with identical parameters)

If the code is correct, all three must agree to high precision.
"""

import numpy as np
import pandas as pd
import pytest

import laytracer


# ── Model parameters (shared by all three approaches) ──────────────────

VP = 5000.0          # P-wave velocity  (m/s)
VS = VP / 1.732      # S-wave velocity   (m/s)
RHO = 2700.0         # density           (kg/m³)
QP = 500.0           # P-wave quality factor
QS = 250.0           # S-wave quality factor

# Geometry
SRC = np.array([0.0, 0.0, 500.0])       # source  (x, y, z) — 500 m depth
RCV = np.array([5000.0, 0.0, 2500.0])   # receiver (x, y, z) — 2500 m depth

EPIC = np.sqrt((RCV[0] - SRC[0]) ** 2 + (RCV[1] - SRC[1]) ** 2)  # 5000 m
DZ = abs(RCV[2] - SRC[2])                                          # 2000 m
DIST = np.sqrt(EPIC ** 2 + DZ ** 2)                                # ~5385 m


# ── Model DataFrames ───────────────────────────────────────────────────

def _homo_model() -> pd.DataFrame:
    """Single-layer (homogeneous) model."""
    return pd.DataFrame({
        "Depth": [0.0],
        "Vp":    [VP],
        "Vs":    [VS],
        "Rho":   [RHO],
        "Qp":    [QP],
        "Qs":    [QS],
    })


def _two_layer_identical_model() -> pd.DataFrame:
    """Two-layer model where both layers have exactly the same properties."""
    return pd.DataFrame({
        "Depth": [0.0, 1500.0],
        "Vp":    [VP, VP],
        "Vs":    [VS, VS],
        "Rho":   [RHO, RHO],
        "Qp":    [QP, QP],
        "Qs":    [QS, QS],
    })


# ── (a) Analytical homogeneous formulas ───────────────────────────────

def _analytical_solution():
    """Closed-form quantities for a straight ray in a homogeneous medium.

    Returns
    -------
    dict with keys: travel_time, tstar, spreading, trans_product, combined
    """
    # Straight-ray distance
    R = DIST

    # Travel time
    tt = R / VP

    # t* (attenuation operator)
    tstar = tt / QP

    # Ray parameter  p = sin(θ) / V = X / (V * R)
    p = EPIC / (VP * R)

    # Geometrical spreading (relative, Červený convention used in LayTracer)
    #   For a homogeneous medium: L = R * V
    spreading = R * VP

    # Transmission coefficient product (no interfaces → 1.0)
    trans_product = 1.0

    # Combined amplitude factor  (T / L) * exp(-pi * f * tstar)
    # We store the deterministic part: trans_product / spreading
    # (frequency-dependent exp term excluded)
    combined = trans_product / spreading

    return dict(
        travel_time=tt,
        ray_parameter=p,
        tstar=tstar,
        spreading=spreading,
        trans_product=trans_product,
        combined=combined,
    )


# ── (b) & (c) Code-based solutions ────────────────────────────────────

def _run_code(vel_df: pd.DataFrame):
    """Trace one ray through *vel_df* and return the same dict as _analytical_solution."""
    result = laytracer.trace_rays(
        sources=SRC,
        receivers=RCV,
        velocity_df=vel_df,
        source_phase="P",
        compute_amplitude=True,
        transcoef_method="standard",
    )
    tt = float(result.travel_times[0])
    p = float(result.ray_parameters[0])
    tstar = float(result.tstar[0])
    spreading = float(result.spreading[0])
    trans_product = float(result.trans_product[0])
    combined = trans_product / spreading

    return dict(
        travel_time=tt,
        ray_parameter=p,
        tstar=tstar,
        spreading=spreading,
        trans_product=trans_product,
        combined=combined,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Test class
# ═══════════════════════════════════════════════════════════════════════

class TestHomogeneousEquivalence:
    """All three approaches must agree on every computed quantity."""

    # ── fixtures ──────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def analytical(self):
        return _analytical_solution()

    @pytest.fixture(scope="class")
    def homo_code(self):
        return _run_code(_homo_model())

    @pytest.fixture(scope="class")
    def layered_code(self):
        return _run_code(_two_layer_identical_model())

    # ── travel time ───────────────────────────────────────────────────

    def test_travel_time_analytical_vs_homo(self, analytical, homo_code):
        """(a) vs (b): travel time."""
        assert homo_code["travel_time"] == pytest.approx(
            analytical["travel_time"], rel=1e-6
        )

    def test_travel_time_analytical_vs_layered(self, analytical, layered_code):
        """(a) vs (c): travel time."""
        assert layered_code["travel_time"] == pytest.approx(
            analytical["travel_time"], rel=1e-6
        )

    def test_travel_time_homo_vs_layered(self, homo_code, layered_code):
        """(b) vs (c): travel time."""
        assert layered_code["travel_time"] == pytest.approx(
            homo_code["travel_time"], rel=1e-6
        )

    # ── ray parameter ─────────────────────────────────────────────────

    def test_ray_parameter_analytical_vs_homo(self, analytical, homo_code):
        """(a) vs (b): ray parameter."""
        assert homo_code["ray_parameter"] == pytest.approx(
            analytical["ray_parameter"], rel=1e-6
        )

    def test_ray_parameter_analytical_vs_layered(self, analytical, layered_code):
        """(a) vs (c): ray parameter."""
        assert layered_code["ray_parameter"] == pytest.approx(
            analytical["ray_parameter"], rel=1e-6
        )

    # ── t* ────────────────────────────────────────────────────────────

    def test_tstar_analytical_vs_homo(self, analytical, homo_code):
        """(a) vs (b): t*."""
        assert homo_code["tstar"] == pytest.approx(
            analytical["tstar"], rel=1e-6
        )

    def test_tstar_analytical_vs_layered(self, analytical, layered_code):
        """(a) vs (c): t*."""
        assert layered_code["tstar"] == pytest.approx(
            analytical["tstar"], rel=1e-6
        )

    def test_tstar_homo_vs_layered(self, homo_code, layered_code):
        """(b) vs (c): t*."""
        assert layered_code["tstar"] == pytest.approx(
            homo_code["tstar"], rel=1e-6
        )

    # ── geometrical spreading ─────────────────────────────────────────

    def test_spreading_analytical_vs_homo(self, analytical, homo_code):
        """(a) vs (b): geometrical spreading."""
        assert homo_code["spreading"] == pytest.approx(
            analytical["spreading"], rel=1e-6
        )

    def test_spreading_analytical_vs_layered(self, analytical, layered_code):
        """(a) vs (c): geometrical spreading."""
        assert layered_code["spreading"] == pytest.approx(
            analytical["spreading"], rel=1e-6
        )

    def test_spreading_homo_vs_layered(self, homo_code, layered_code):
        """(b) vs (c): geometrical spreading."""
        assert layered_code["spreading"] == pytest.approx(
            homo_code["spreading"], rel=1e-6
        )

    # ── transmission coefficient product ──────────────────────────────

    def test_trans_product_analytical_vs_homo(self, analytical, homo_code):
        """(a) vs (b): transmission coefficient product."""
        assert homo_code["trans_product"] == pytest.approx(
            analytical["trans_product"], rel=1e-6
        )

    def test_trans_product_analytical_vs_layered(self, analytical, layered_code):
        """(a) vs (c): transmission coefficient product = 1.0 (identical layers)."""
        assert layered_code["trans_product"] == pytest.approx(
            analytical["trans_product"], rel=1e-6
        )

    def test_trans_product_homo_vs_layered(self, homo_code, layered_code):
        """(b) vs (c): transmission coefficient product."""
        assert layered_code["trans_product"] == pytest.approx(
            homo_code["trans_product"], rel=1e-6
        )

    # ── combined amplitude factor (T / L) ─────────────────────────────

    def test_combined_analytical_vs_homo(self, analytical, homo_code):
        """(a) vs (b): combined deterministic amplitude factor."""
        assert homo_code["combined"] == pytest.approx(
            analytical["combined"], rel=1e-6
        )

    def test_combined_analytical_vs_layered(self, analytical, layered_code):
        """(a) vs (c): combined deterministic amplitude factor."""
        assert layered_code["combined"] == pytest.approx(
            analytical["combined"], rel=1e-6
        )

    def test_combined_homo_vs_layered(self, homo_code, layered_code):
        """(b) vs (c): combined deterministic amplitude factor."""
        assert layered_code["combined"] == pytest.approx(
            homo_code["combined"], rel=1e-6
        )

    # ── summary print (always runs last) ──────────────────────────────

    def test_zz_summary(self, analytical, homo_code, layered_code):
        """Print a comparison table (always passes)."""
        keys = ["travel_time", "ray_parameter", "tstar",
                "spreading", "trans_product", "combined"]
        header = f"{'Quantity':<20s} {'Analytical':>14s} {'Homo code':>14s} {'Layered code':>14s}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for k in keys:
            lines.append(
                f"{k:<20s} {analytical[k]:>14.8g} {homo_code[k]:>14.8g} {layered_code[k]:>14.8g}"
            )
        lines.append(sep)
        print("\n" + "\n".join(lines))
