r"""Tests for the core solver: offset equation, Newton iteration, and solve()."""

import numpy as np
import pandas as pd
import pytest

import laytracer


# ── Helpers ──

def _simple_model():
    """Three-layer model for testing."""
    return pd.DataFrame({
        "Depth": [0.0, 1000.0, 2000.0],
        "Vp":    [3000.0, 4500.0, 6000.0],
        "Vs":    [1500.0, 2250.0, 3000.0],
        "Rho":   [2200.0, 2500.0, 2800.0],
        "Qp":    [200.0,  400.0,  600.0],
        "Qs":    [100.0,  200.0,  300.0],
    })


def _homo_model(v=5000.0, depth=5000.0):
    """Homogeneous half-space."""
    return pd.DataFrame({
        "Depth": [0.0],
        "Vp":    [v],
        "Vs":    [v / 1.732],
        "Rho":   [2700.0],
        "Qp":    [500.0],
        "Qs":    [250.0],
    })


# ═══════════════════════════════════════════════════════════════════════
#  Offset equation
# ═══════════════════════════════════════════════════════════════════════

class TestOffset:
    def test_zero_q(self):
        """X(0) = 0 for any layer configuration."""
        h = np.array([500.0, 1000.0, 500.0])
        lmd = np.array([0.5, 0.75, 1.0])
        assert laytracer.offset(0.0, h, lmd) == pytest.approx(0.0, abs=1e-12)

    def test_monotonic(self):
        """X(q) increases monotonically with q."""
        h = np.array([500.0, 1000.0, 500.0])
        lmd = np.array([0.5, 0.75, 1.0])
        q_vals = np.linspace(0.01, 100, 200)
        X_vals = [laytracer.offset(q, h, lmd) for q in q_vals]
        for i in range(1, len(X_vals)):
            assert X_vals[i] > X_vals[i - 1]

    def test_derivative_numerical(self):
        """dX/dq matches finite-difference approximation."""
        h = np.array([300.0, 700.0, 1000.0])
        lmd = np.array([0.6, 0.8, 1.0])
        q = 5.0
        eps = 1e-6
        fd = (laytracer.offset(q + eps, h, lmd) - laytracer.offset(q - eps, h, lmd)) / (2 * eps)
        analytical = laytracer.offset_dq(q, h, lmd)
        assert analytical == pytest.approx(fd, rel=1e-5)

    def test_second_derivative_numerical(self):
        """d²X/dq² matches finite-difference approximation."""
        h = np.array([300.0, 700.0, 1000.0])
        lmd = np.array([0.6, 0.8, 1.0])
        q = 5.0
        eps = 1e-5
        fd = (
            laytracer.offset_dq(q + eps, h, lmd) - laytracer.offset_dq(q - eps, h, lmd)
        ) / (2 * eps)
        analytical = laytracer.offset_dq2(q, h, lmd)
        assert analytical == pytest.approx(fd, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
#  Parameter conversions
# ═══════════════════════════════════════════════════════════════════════

class TestConversions:
    def test_roundtrip(self):
        """q → p → q roundtrip."""
        vmax = 6000.0
        for q in [0.001, 0.1, 1.0, 10.0, 100.0]:
            p = laytracer.p_from_q(q, vmax)
            q2 = laytracer.q_from_p(p, vmax)
            assert q2 == pytest.approx(q, rel=1e-10)

    def test_p_limits(self):
        """p → 0 as q → 0; p → 1/vmax as q → ∞."""
        vmax = 5000.0
        assert laytracer.p_from_q(1e-10, vmax) == pytest.approx(0.0, abs=1e-12)
        p_large = laytracer.p_from_q(1e8, vmax)
        assert p_large == pytest.approx(1.0 / vmax, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
#  Newton convergence
# ═══════════════════════════════════════════════════════════════════════

class TestNewton:
    def test_convergence(self):
        """Newton iteration converges within 5 steps for typical case."""
        h = np.array([500.0, 1000.0, 500.0])
        lmd = np.array([0.5, 0.75, 1.0])
        X_target = 3000.0
        q = laytracer.initial_q(X_target, h, lmd)
        for _ in range(5):
            q, X_new = laytracer.newton_step(q, X_target, h, lmd)
        assert X_new == pytest.approx(X_target, abs=0.1)

    def test_near_field(self):
        """Convergence for short-offset ray."""
        h = np.array([500.0, 1000.0, 500.0])
        lmd = np.array([0.5, 0.75, 1.0])
        X_target = 100.0
        q = laytracer.initial_q(X_target, h, lmd)
        for _ in range(5):
            q, X_new = laytracer.newton_step(q, X_target, h, lmd)
        assert X_new == pytest.approx(X_target, abs=0.1)

    def test_far_field(self):
        """Convergence for large-offset ray."""
        h = np.array([500.0, 1000.0, 500.0])
        lmd = np.array([0.5, 0.75, 1.0])
        X_target = 50000.0
        q = laytracer.initial_q(X_target, h, lmd)
        for _ in range(10):
            q, X_new = laytracer.newton_step(q, X_target, h, lmd)
        assert X_new == pytest.approx(X_target, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════════
#  solve()
# ═══════════════════════════════════════════════════════════════════════

class TestSolve:
    def test_homogeneous(self):
        """Homogeneous medium: tt = distance / velocity."""
        df = _homo_model(v=5000.0)
        stack = laytracer.build_layer_stack(df, z_src=0.0, z_rcv=3000.0)
        epic = 4000.0
        res = laytracer.solve(stack, epic, z_src=0.0, z_rcv=3000.0)
        dist = np.sqrt(epic**2 + 3000.0**2)
        assert res.travel_time == pytest.approx(dist / 5000.0, rel=1e-4)

    def test_vertical_ray(self):
        """Vertical ray: tt = Σ h_k / v_k."""
        df = _simple_model()
        stack = laytracer.build_layer_stack(df, z_src=0.0, z_rcv=2500.0)
        res = laytracer.solve(stack, epicentral_dist=0.0, z_src=0.0, z_rcv=2500.0)
        h = stack.h
        v = stack.vp
        expected_tt = np.sum(h / v)
        assert res.travel_time == pytest.approx(expected_tt, rel=1e-6)

    def test_ray_endpoints(self):
        """Ray starts at source and ends at receiver coordinates."""
        df = _simple_model()
        stack = laytracer.build_layer_stack(df, z_src=500.0, z_rcv=2500.0)
        epic = 5000.0
        res = laytracer.solve(stack, epic, z_src=500.0, z_rcv=2500.0)
        # Start
        assert res.ray_path[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert res.ray_path[0, 1] == pytest.approx(500.0, abs=1e-6)
        # End
        assert res.ray_path[-1, 0] == pytest.approx(epic, rel=1e-3)
        assert res.ray_path[-1, 1] == pytest.approx(2500.0, abs=1e-3)

    def test_travel_time_positive(self):
        """Travel time is always positive."""
        df = _simple_model()
        stack = laytracer.build_layer_stack(df, z_src=200.0, z_rcv=2800.0)
        for epic in [100.0, 1000.0, 10000.0, 50000.0]:
            res = laytracer.solve(stack, epic, z_src=200.0, z_rcv=2800.0)
            assert res.travel_time > 0

    def test_ray_parameter_positive(self):
        """Ray parameter p is non-negative."""
        df = _simple_model()
        stack = laytracer.build_layer_stack(df, z_src=500.0, z_rcv=1500.0)
        res = laytracer.solve(stack, epicentral_dist=3000.0, z_src=500.0, z_rcv=1500.0)
        assert res.ray_parameter >= 0
