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
        X_vals = np.array([laytracer.offset(q, h, lmd) for q in q_vals])
        # Bulletproof monotonicity check: ensure consecutive values grow by at least 1e-12
        assert np.all(np.diff(X_vals) >= 1e-12)

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
        """d²X/dq² matches central difference of dX/dq."""
        h = np.array([300.0, 700.0, 1000.0])
        lmd = np.array([0.6, 0.8, 1.0])
        q = 5.0
        eps = 1e-5
        fd = (
            laytracer.offset_dq(q + eps, h, lmd) - laytracer.offset_dq(q - eps, h, lmd)
        ) / (2 * eps)
        analytical = laytracer.offset_dq2(q, h, lmd)
        assert analytical == pytest.approx(fd, rel=1e-3)

    def test_second_derivative_from_offset(self):
        """d²X/dq² matches central second difference of X."""
        h = np.array([300.0, 700.0, 1000.0])
        lmd = np.array([0.6, 0.8, 1.0])
        q = 5.0
        eps = 1e-4
        fd2 = (
            laytracer.offset(q + eps, h, lmd) 
            - 2 * laytracer.offset(q, h, lmd) 
            + laytracer.offset(q - eps, h, lmd)
        ) / (eps**2)
        analytical = laytracer.offset_dq2(q, h, lmd)
        assert analytical == pytest.approx(fd2, rel=1e-3)


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
        
        # Manually construct args for solve
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": 0.0, "end_z": 3000.0
        }]
        
        res = laytracer.solve(h, v, segments, [], epic, z_src=0.0, z_rcv=3000.0)
        dist = np.sqrt(epic**2 + 3000.0**2)
        assert res.travel_time == pytest.approx(dist / 5000.0, rel=1e-4)

    def test_vertical_ray(self):
        """Vertical ray: tt = Σ h_k / v_k."""
        df = _simple_model()
        stack = laytracer.build_layer_stack(df, z_src=0.0, z_rcv=2500.0)
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": 0.0, "end_z": 2500.0
        }]
        
        res = laytracer.solve(h, v, segments, [], 0.0, z_src=0.0, z_rcv=2500.0)
        expected_tt = np.sum(h / v)
        assert res.travel_time == pytest.approx(expected_tt, rel=1e-6)

    def test_ray_endpoints(self):
        """Ray starts at source and ends at receiver coordinates."""
        df = _simple_model()
        z_src, z_rcv = 500.0, 2500.0
        stack = laytracer.build_layer_stack(df, z_src=z_src, z_rcv=z_rcv)
        epic = 5000.0
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": z_src, "end_z": z_rcv
        }]
        
        res = laytracer.solve(h, v, segments, [], epic, z_src=z_src, z_rcv=z_rcv)
        # Start
        assert res.ray_path[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert res.ray_path[0, 1] == pytest.approx(500.0, abs=1e-6)
        # End
        assert res.ray_path[-1, 0] == pytest.approx(epic, rel=1e-3)
        assert res.ray_path[-1, 1] == pytest.approx(2500.0, abs=1e-3)

    def test_travel_time_positive(self):
        """Travel time is always positive."""
        df = _simple_model()
        z_src, z_rcv = 200.0, 2800.0
        stack = laytracer.build_layer_stack(df, z_src=z_src, z_rcv=z_rcv)
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": z_src, "end_z": z_rcv
        }]
        
        for epic in [100.0, 1000.0, 10000.0, 50000.0]:
            res = laytracer.solve(h, v, segments, [], epic, z_src=z_src, z_rcv=z_rcv)
            assert res.travel_time > 0

    def test_ray_parameter_positive(self):
        """Ray parameter p is non-negative."""
        df = _simple_model()
        z_src, z_rcv = 500.0, 1500.0
        stack = laytracer.build_layer_stack(df, z_src=z_src, z_rcv=z_rcv)
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": z_src, "end_z": z_rcv
        }]
        
        res = laytracer.solve(h, v, segments, [], 3000.0, z_src=z_src, z_rcv=z_rcv)
        assert res.ray_parameter >= 0

    def test_snell_law_upward_ray(self):
        """Snell's law: sin(theta_k)/v_k = p is constant across layers for upward ray."""
        df = _simple_model()
        z_src, z_rcv = 2500.0, 0.0
        stack = laytracer.build_layer_stack(df, z_src=z_src, z_rcv=z_rcv)
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": z_src, "end_z": z_rcv
        }]
        
        res = laytracer.solve(h, v, segments, [], 5000.0, z_src=z_src, z_rcv=z_rcv)
        p = res.ray_parameter

        # Verify from ray geometry: sin(theta_k)/v_k should equal p
        for k in range(stack.n_layers):
            x0, z0 = res.ray_path[k]
            x1, z1 = res.ray_path[k + 1]
            dx = abs(x1 - x0)
            dz = abs(z1 - z0)
            seg_len = np.sqrt(dx**2 + dz**2)
            sin_theta = dx / seg_len
            # Which velocity does this segment use?
            # Upward ray: segment 0 is deepest, so idx = N-1-k
            idx = stack.n_layers - 1 - k
            p_from_geom = sin_theta / stack.vp[idx]
            assert p_from_geom == pytest.approx(p, rel=1e-3), (
                f"Segment {k} (layer idx {idx}): p_geom={p_from_geom:.6e}, "
                f"p_solver={p:.6e}"
            )

    def test_ray_angle_steeper_in_slow_layer(self):
        """Ray is steeper (smaller angle from vertical) in slower layers."""
        df = _simple_model()
        z_src, z_rcv = 2500.0, 0.0
        # Upward ray: source deep, receiver shallow
        stack = laytracer.build_layer_stack(df, z_src=z_src, z_rcv=z_rcv)
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": z_src, "end_z": z_rcv
        }]
        
        res = laytracer.solve(h, v, segments, [], 5000.0, z_src=z_src, z_rcv=z_rcv)

        # Compute dx/dz for each segment from ray path
        # The ray goes upward, so layer order in path is: deepest first
        ratios = []
        for k in range(stack.n_layers):
            x0, z0 = res.ray_path[k]
            x1, z1 = res.ray_path[k + 1]
            dx = abs(x1 - x0)
            dz = abs(z1 - z0)
            ratios.append(dx / dz)  # larger = flatter = faster layer

        # Ratios should increase from first segment (deepest=fastest)
        # to last segment (shallowest=slowest)... wait no:
        # Deepest layer has highest velocity → flatter ray → larger ratio
        # Shallowest layer has lowest velocity → steeper ray → smaller ratio
        # Since upward ray: segment 0 is deepest (fastest), last is shallowest (slowest)
        assert ratios[0] > ratios[-1], (
            f"Ray should be flatter in fast deep layer (dx/dz={ratios[0]:.3f}) "
            f"than in slow shallow layer (dx/dz={ratios[-1]:.3f})"
        )
