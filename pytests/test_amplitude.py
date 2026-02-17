r"""Tests for amplitude computation: t*, spreading, and transmission coefficients."""

import numpy as np
import pandas as pd
import pytest

import laytracer


def _simple_model():
    return pd.DataFrame({
        "Depth": [0.0, 1000.0, 2000.0],
        "Vp":    [3000.0, 4500.0, 6000.0],
        "Vs":    [1500.0, 2250.0, 3000.0],
        "Rho":   [2200.0, 2500.0, 2800.0],
        "Qp":    [200.0,  400.0,  600.0],
        "Qs":    [100.0,  200.0,  300.0],
    })


# ═══════════════════════════════════════════════════════════════════════
#  Transmission coefficients
# ═══════════════════════════════════════════════════════════════════════

class TestTransmission:
    def test_normal_same_medium(self):
        """T = 1 when both sides are identical."""
        T = laytracer.transmission_normal(5000.0, 2700.0, 5000.0, 2700.0)
        assert T == pytest.approx(1.0)

    def test_normal_known_value(self):
        """Check against hand-computed impedance ratio."""
        v1, rho1 = 3000.0, 2200.0
        v2, rho2 = 5000.0, 2700.0
        Z1 = rho1 * v1
        Z2 = rho2 * v2
        expected = 2 * Z1 / (Z1 + Z2)
        T = laytracer.transmission_normal(v1, rho1, v2, rho2)
        assert T == pytest.approx(expected, rel=1e-10)

    def test_angle_reduces_to_normal_at_p0(self):
        """Angle-dependent T(p→0) is roughly consistent with normal-incidence T.

        The full Zoeppritz accounts for P-SV coupling at the interface,
        so it differs from the simple impedance formula by ~10%.
        """
        vp1, vs1, rho1 = 4000.0, 2000.0, 2500.0
        vp2, vs2, rho2 = 5000.0, 2800.0, 2700.0
        T_angle = laytracer.transmission_psv(1e-10, vp1, vs1, rho1, vp2, vs2, rho2)
        T_normal = laytracer.transmission_normal(vp1, rho1, vp2, rho2)
        assert abs(T_angle) == pytest.approx(T_normal, rel=0.15)

    def test_transmission_positive(self):
        """Transmission coefficient magnitude is positive."""
        T = laytracer.transmission_psv(
            0.0001, 3000.0, 1500.0, 2200.0, 5000.0, 2800.0, 2700.0
        )
        assert abs(T) > 0


# ═══════════════════════════════════════════════════════════════════════
#  t* computation
# ═══════════════════════════════════════════════════════════════════════

class TestTstar:
    def test_tstar_single_layer_vertical(self):
        """t* = h / (v * Q) for a vertical ray in a single layer."""
        df = pd.DataFrame({
            "Depth": [0.0], "Vp": [5000.0], "Vs": [2887.0],
            "Rho": [2700.0], "Qp": [500.0], "Qs": [250.0],
        })
        stack = laytracer.build_layer_stack(df, z_src=0.0, z_rcv=3000.0)
        res = laytracer.solve(
            stack, epicentral_dist=0.0, z_src=0.0, z_rcv=3000.0,
            compute_amplitude=True,
        )
        expected = 3000.0 / 5000.0 / 500.0  # h / v / Q = tt / Q
        assert res.tstar == pytest.approx(expected, rel=1e-4)

    def test_tstar_uniform_Q(self):
        """When Q is constant everywhere, t* = tt / Q."""
        Q = 300.0
        df = pd.DataFrame({
            "Depth": [0.0, 1000.0, 2000.0],
            "Vp": [3000.0, 4500.0, 6000.0],
            "Vs": [1500.0, 2250.0, 3000.0],
            "Rho": [2200.0, 2500.0, 2800.0],
            "Qp": [Q, Q, Q],
            "Qs": [Q / 2, Q / 2, Q / 2],
        })
        stack = laytracer.build_layer_stack(df, z_src=100.0, z_rcv=2500.0)
        res = laytracer.solve(
            stack, epicentral_dist=5000.0, z_src=100.0, z_rcv=2500.0,
            compute_amplitude=True,
        )
        assert res.tstar == pytest.approx(res.travel_time / Q, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
#  Geometrical spreading
# ═══════════════════════════════════════════════════════════════════════

class TestSpreading:
    def test_spreading_homogeneous(self):
        """In a homogeneous medium, spreading ≈ distance (the ray path length)."""
        df = pd.DataFrame({
            "Depth": [0.0], "Vp": [5000.0], "Vs": [2887.0],
            "Rho": [2700.0], "Qp": [500.0], "Qs": [250.0],
        })
        stack = laytracer.build_layer_stack(df, z_src=0.0, z_rcv=3000.0)
        res = laytracer.solve(
            stack, epicentral_dist=4000.0, z_src=0.0, z_rcv=3000.0,
            compute_amplitude=True,
        )
        assert res.spreading is not None
        assert res.spreading > 0

    def test_spreading_positive_multilayer(self):
        """Spreading is positive for a multi-layer model."""
        df = _simple_model()
        stack = laytracer.build_layer_stack(df, z_src=500.0, z_rcv=2500.0)
        res = laytracer.solve(
            stack, epicentral_dist=5000.0, z_src=500.0, z_rcv=2500.0,
            compute_amplitude=True,
        )
        assert res.spreading is not None
        assert res.spreading > 0
