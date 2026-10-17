r"""Tests for the core solver: offset equation, Newton iteration, and solve()."""

import decimal
import inspect
import warnings
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

import laytracer


def test_solve_additive_arguments_follow_the_legacy_positional_contract():
    """New controls are trailing so 0.4 positional calls retain their meaning."""
    names = list(inspect.signature(laytracer.solve).parameters)
    assert names[:15] == [
        "h",
        "v",
        "segments",
        "interactions",
        "epicentral_dist",
        "z_src",
        "z_rcv",
        "requested",
        "return_ray_path",
        "need_ray_parameter",
        "need_tstar",
        "need_spreading",
        "need_trans_product",
        "transcoef_method",
        "tol",
    ]
    assert names[15:] == [
        "max_iter",
        "need_complex_coefficient_product",
        "need_diagnostics",
    ]


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


def _grazing_model():
    """Three-layer model whose half-space is the fastest layer."""
    return pd.DataFrame({
        "Depth": [0.0, 1000.0, 2000.0],
        "Vp":    [1500.0, 5000.0, 7000.0],
        "Vs":    [857.0, 2857.0, 4000.0],
        "Rho":   [2000.0, 2400.0, 2700.0],
        "Qp":    [100.0, 200.0, 400.0],
        "Qs":    [50.0, 100.0, 200.0],
    })


def _direct_solve(df, z_src, z_rcv, offset, **kwargs):
    """Solve one direct P ray through the layer stack between two depths."""
    stack = laytracer.build_layer_stack(df, z_src=z_src, z_rcv=z_rcv)
    segments = [{
        "h": stack.h, "v": stack.vp, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
        "qp": stack.qp, "qs": stack.qs, "phase": "P",
        "start_z": z_src, "end_z": z_rcv
    }]
    res = laytracer.solve(
        stack.h, stack.vp, segments, [], offset, z_src=z_src, z_rcv=z_rcv, **kwargs
    )
    return res, stack


def _q_reference(h, v, offset):
    """Cancellation-free direct ray: solve X(q) = offset, evaluate the legs in q.

    Returns q, the leg cosines and the relative geometrical spreading.
    """
    vmax = v.max()
    lmd = v / vmax
    s = 1.0 - lmd**2
    q = brentq(
        lambda q: np.sum(h * lmd * q / np.sqrt(1.0 + s * q * q)) - offset,
        0.0, 1e15, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500,
    )
    c = np.sqrt((1.0 + s * q * q) / (1.0 + q * q))
    p = q / (vmax * np.sqrt(1.0 + q * q))
    dxdp = np.sum(lmd * h / (1.0 + s * q * q) ** 1.5) * vmax * (1.0 + q * q) ** 1.5
    spreading = np.sqrt(offset * dxdp * c[0] * c[-1] / p)
    return q, c, spreading


def _margin_reference(x, z=1.0):
    """1 - x / sqrt(x**2 + z**2), evaluated as written in 50-digit arithmetic."""
    with decimal.localcontext() as ctx:
        ctx.prec = 50
        x, z = decimal.Decimal(x), decimal.Decimal(z)
        return float(1 - x / (x * x + z * z).sqrt())





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
        for q in [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
            p = laytracer.p_from_q(q, vmax)
            q2 = laytracer.q_from_p(p, vmax)
            assert q2 == pytest.approx(q, rel=1e-5)

    def test_p_limits(self):
        """p → 0 as q → 0; p → 1/vmax as q → ∞."""
        vmax = 5000.0
        assert laytracer.p_from_q(1e-10, vmax) == pytest.approx(0.0, abs=1e-10)
        p_large = laytracer.p_from_q(1e8, vmax)
        assert p_large == pytest.approx(1.0 / vmax, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
#  Newton convergence
# ═══════════════════════════════════════════════════════════════════════

class TestNewton:
    def test_convergence(self):
        """Newton iteration converges to tolerance within a reasonable range of steps."""
        h = np.array([500.0, 1000.0, 500.0])
        lmd = np.array([0.5, 0.75, 1.0])
        X_target = 3000.0
        q = laytracer.initial_q(X_target, h, lmd)
        
        residuals = [abs(laytracer.offset(q, h, lmd) - X_target)]
        for _ in range(10):
            q, X_new = laytracer.newton_step(q, X_target, h, lmd)
            residuals.append(abs(X_new - X_target))
        
        # Should reach low residual
        assert min(residuals) < 0.1
        # Should generally be decreasing (monotone reduction check)
        # Newton can fluctuate if the initial guess is poor, but it should eventually settle.
        # For this convex problem, it's usually monotonic.
        assert residuals[-1] < residuals[0]

    def test_near_field(self):
        """Convergence for short-offset ray."""
        h = np.array([500.0, 1000.0, 500.0])
        lmd = np.array([0.5, 0.75, 1.0])
        X_target = 100.0
        q = laytracer.initial_q(X_target, h, lmd)
        
        residuals = [abs(laytracer.offset(q, h, lmd) - X_target)]
        for _ in range(10):
            q, X_new = laytracer.newton_step(q, X_target, h, lmd)
            residuals.append(abs(X_new - X_target))
            
        assert min(residuals) < 0.1
        assert residuals[-1] < residuals[0]

    def test_far_field(self):
        """Convergence for large-offset ray."""
        h = np.array([500.0, 1000.0, 500.0])
        lmd = np.array([0.5, 0.75, 1.0])
        X_target = 50000.0
        q = laytracer.initial_q(X_target, h, lmd)
        
        residuals = [abs(laytracer.offset(q, h, lmd) - X_target)]
        for _ in range(15):
            q, X_new = laytracer.newton_step(q, X_target, h, lmd)
            residuals.append(abs(X_new - X_target))
            
        assert min(residuals) < 1e-2 * X_target
        assert residuals[-1] < residuals[0]

    def test_far_field_quadratic_root_does_not_cancel(self):
        """A far-field step still moves when 4|AC| is below eps * B**2."""
        h = np.array([1000.0, 1000.0, 100.0])
        lmd = np.array([1500.0, 5000.0, 7000.0]) / 7000.0
        X_target = 105000.0
        # Iterate at which the textbook root formula returned a zero step
        q_stalled = 1037.59998733142
        assert laytracer.offset(q_stalled, h, lmd) - X_target > 1e-4

        _, X_new = laytracer.newton_step(q_stalled, X_target, h, lmd)
        assert abs(X_new - X_target) < 1e-9


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
        # Structure and shape
        assert res.ray_path.shape[1] == 2
        
        # Start
        assert res.ray_path[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert res.ray_path[0, 1] == pytest.approx(500.0, abs=1e-6)
        # End
        assert res.ray_path[-1, 0] == pytest.approx(epic, rel=1e-3)
        assert res.ray_path[-1, 1] == pytest.approx(2500.0, abs=1e-3)

        # Monotonicity in X
        assert np.all(np.diff(res.ray_path[:, 0]) >= 0)

    def test_solve_fallback(self):
        """solve() certifies a bracketed root when Newton is disabled."""
        df = _simple_model()
        z_src, z_rcv = 0.0, 2500.0
        stack = laytracer.build_layer_stack(df, z_src=z_src, z_rcv=z_rcv)
        epic = 5000.0
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": z_src, "end_z": z_rcv
        }]
        
        res = laytracer.solve(
            h,
            v,
            segments,
            [],
            epic,
            z_src=z_src,
            z_rcv=z_rcv,
            requested={"travel_times", "rays", "ray_parameters", "diagnostics"},
            max_iter=0,
        )
        assert res.ray_path[-1, 0] == pytest.approx(epic, rel=1e-3)
        assert res.diagnostics.converged
        assert res.diagnostics.method == "q_brentq"
        assert res.diagnostics.iterations == 0
        assert res.diagnostics.absolute_offset_residual < 1e-4

    def test_newton_diagnostics_recompute_offset(self):
        """The reported certificate agrees with an independent X(q) call."""
        df = _simple_model()
        stack = laytracer.build_layer_stack(df, z_src=200.0, z_rcv=2800.0)
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs,
            "rho": stack.rho, "qp": stack.qp, "qs": stack.qs,
            "phase": "P", "start_z": 200.0, "end_z": 2800.0,
        }]
        target = 7000.0
        res = laytracer.solve(
            h,
            v,
            segments,
            [],
            target,
            z_src=200.0,
            z_rcv=2800.0,
            requested={"travel_times", "ray_parameters", "diagnostics"},
        )

        diag = res.diagnostics
        independent_residual = laytracer.offset(diag.final_q, h, v / v.max()) - target
        assert diag.converged
        assert diag.method in {"q_newton", "q_brentq"}
        assert diag.signed_offset_residual == pytest.approx(independent_residual, abs=1e-12)
        assert diag.absolute_offset_residual == pytest.approx(abs(independent_residual))
        assert diag.criticality_margin > 0.0
        assert np.isfinite(diag.conditioning)

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

        # Reconstruct the velocities used in the order of path points
        v_used = []
        for seg in segments:
            n_lay = len(seg["h"])
            going_down = seg["end_z"] >= seg["start_z"]
            indices = range(n_lay) if going_down else range(n_lay - 1, -1, -1)
            for k in indices:
                v_used.append(seg["v"][k])

        # Verify from ray geometry: sin(theta_k)/v_k should equal p
        for k in range(len(v_used)):
            x0, z0 = res.ray_path[k]
            x1, z1 = res.ray_path[k + 1]
            dx = abs(x1 - x0)
            dz = abs(z1 - z0)
            seg_len = np.sqrt(dx**2 + dz**2)
            sin_theta = dx / seg_len
            
            p_from_geom = sin_theta / v_used[k]
            assert p_from_geom == pytest.approx(p, rel=1e-3), (
                f"Segment {k}: p_geom={p_from_geom:.6e}, "
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
        n_segments = len(res.ray_path) - 1
        for k in range(n_segments):
            x0, z0 = res.ray_path[k]
            x1, z1 = res.ray_path[k + 1]
            dx = abs(x1 - x0)
            dz = abs(z1 - z0)
            ratios.append(dx / dz)  # larger = flatter = faster layer
        
        assert ratios[0] > ratios[-1], (
            f"Ray should be flatter in fast deep layer (dx/dz={ratios[0]:.3f}) "
            f"than in slow shallow layer (dx/dz={ratios[-1]:.3f})"
        )

    @pytest.mark.parametrize("tol", [1e-4, 1e-6])
    def test_far_field_converges_without_fallback(self, tol):
        """A ray at 50 times the source depth converges in the quadratic step."""
        df = pd.DataFrame({
            "Depth": [0.0, 1000.0, 2000.0],
            "Vp":    [1500.0, 5000.0, 7000.0],
            "Vs":    [857.0, 2857.0, 4000.0],
            "Rho":   [2000.0, 2400.0, 2700.0],
            "Qp":    [100.0, 200.0, 400.0],
            "Qs":    [50.0, 100.0, 200.0],
        })
        z_src, z_rcv = 2100.0, 0.0
        stack = laytracer.build_layer_stack(df, z_src=z_src, z_rcv=z_rcv)

        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P",
            "start_z": z_src, "end_z": z_rcv
        }]

        res = laytracer.solve(
            h,
            v,
            segments,
            [],
            105000.0,
            z_src=z_src,
            z_rcv=z_rcv,
            requested={"travel_times", "diagnostics"},
            tol=tol,
        )
        assert res.diagnostics.method == "q_newton"
        assert res.diagnostics.iterations <= 2
        assert res.diagnostics.absolute_offset_residual <= tol

    def test_large_offset_rays_need_no_fallback(self):
        """Seeded rays at 10-100 times the depth converge without the fallback."""
        rng = np.random.default_rng(17)
        for _ in range(200):
            n = int(rng.integers(2, 21))
            h = np.exp(rng.uniform(np.log(20.0), np.log(1000.0), n))
            v = rng.uniform(1500.0, 7000.0, n)
            z = float(np.sum(h))
            segments = [{
                "h": h, "v": v, "vp": v, "vs": v / 1.8, "rho": np.ones(n),
                "qp": None, "qs": None, "phase": "P",
                "start_z": z, "end_z": 0.0,
            }]
            target = 10.0 ** rng.uniform(1.0, 2.0) * z

            res = laytracer.solve(
                h,
                v,
                segments,
                [],
                target,
                z_src=z,
                z_rcv=0.0,
                requested={"travel_times", "diagnostics"},
                tol=1e-6,
            )
            assert res.diagnostics.method == "q_newton", (
                f"fallback for {n} layers at offset/depth {target / z:.1f}"
            )

    def test_grazing_ray_outputs_match_q_evaluation(self):
        """Travel time, t*, spreading and path end stay exact for a grazing ray."""
        offset = 21000.0
        res, stack = _direct_solve(
            _grazing_model(), 2000.1, 0.0, offset,
            requested={"travel_times", "rays", "tstar", "spreading"}, tol=1e-9,
        )
        h, v = stack.h, stack.vp
        q, c, spreading = _q_reference(h, v, offset)
        assert q > 1e5

        assert res.travel_time == pytest.approx(np.sum(h / (v * c)), rel=1e-11)
        assert res.tstar == pytest.approx(np.sum(h / (v * c * stack.qp)), rel=1e-11)
        assert res.spreading == pytest.approx(spreading, rel=1e-11)
        assert res.ray_path[-1, 0] == pytest.approx(offset, abs=1e-8)

    @pytest.mark.parametrize("below", [1e-6, 1e-3, 0.1, 1.0])
    def test_source_just_below_fast_interface(self, below):
        """A source just below a faster interface gives finite, exact travel times."""
        tops = np.arange(0.0, 1801.0, 200.0)
        vp = np.array(
            [1320.5, 1777.0, 2164.0, 2588.5, 2913.0, 3203.5, 3550.5, 3754.0, 3887.0, 5820.0]
        )
        df = pd.DataFrame({"Depth": tops, "Vp": vp, "Vs": vp / 1.75})
        for z_rcv in (15.0, 455.0):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                res, stack = _direct_solve(
                    df, 1800.0 + below, z_rcv, 3000.0,
                    requested={"travel_times"}, tol=1e-9,
                )
            _, c, _ = _q_reference(stack.h, stack.vp, 3000.0)
            assert res.travel_time == pytest.approx(
                np.sum(stack.h / (stack.vp * c)), rel=1e-12
            )

    def test_spreading_returned_at_very_large_q(self):
        """Spreading stays defined when 1 - (p vmax)^2 is far below 1e-15."""
        offset = 21000.0
        res, stack = _direct_solve(
            _grazing_model(), 2000.0 + 1e-4, 0.0, offset,
            requested={"travel_times", "spreading", "diagnostics"}, tol=1e-9,
        )
        _, _, spreading = _q_reference(stack.h, stack.vp, offset)
        assert res.diagnostics.final_q > 1e8
        assert res.spreading == pytest.approx(spreading, rel=1e-11)

    def test_large_offset_travel_times_match_q_evaluation(self):
        """Seeded rays at 10-100 times the depth: exact travel times and path ends."""
        rng = np.random.default_rng(18)
        for _ in range(100):
            n = int(rng.integers(2, 21))
            h = np.exp(rng.uniform(np.log(20.0), np.log(1000.0), n))
            v = rng.uniform(1500.0, 7000.0, n)
            z = float(np.sum(h))
            segments = [{
                "h": h, "v": v, "vp": v, "vs": v / 1.8, "rho": np.ones(n),
                "qp": None, "qs": None, "phase": "P",
                "start_z": z, "end_z": 0.0,
            }]
            target = 10.0 ** rng.uniform(1.0, 2.0) * z

            res = laytracer.solve(
                h,
                v,
                segments,
                [],
                target,
                z_src=z,
                z_rcv=0.0,
                requested={"travel_times", "rays"},
                tol=1e-9,
            )
            _, c, _ = _q_reference(h, v, target)
            assert res.travel_time == pytest.approx(np.sum(h / (v * c)), rel=1e-11)
            assert res.ray_path[-1, 0] == pytest.approx(target, rel=1e-12, abs=1e-8)

    @pytest.mark.parametrize("below", [10.0, 0.1, 1e-3, 1e-4])
    def test_criticality_margin_keeps_precision_near_grazing(self, below):
        """The criticality margin equals 1 - p vmax at the solved q to full precision."""
        res, _ = _direct_solve(
            _grazing_model(), 2000.0 + below, 0.0, 21000.0,
            requested={"travel_times", "diagnostics"}, tol=1e-9,
        )
        margin = res.diagnostics.criticality_margin
        assert margin > 0.0
        assert margin == pytest.approx(
            _margin_reference(res.diagnostics.final_q), rel=1e-13, abs=0.0
        )

    # Binary fractions keep 500 - dz exact, so the stack thickness is dz itself
    @pytest.mark.parametrize("dz", [1.0, 2.0**-7, 2.0**-14])
    def test_single_leg_criticality_margin_keeps_precision(self, dz):
        """A nearly horizontal straight leg reports its exact criticality margin."""
        offset = 10000.0
        res, _ = _direct_solve(
            _homo_model(), 500.0, 500.0 - dz, offset,
            requested={"travel_times", "diagnostics"},
        )
        margin = res.diagnostics.criticality_margin
        assert margin > 0.0
        assert margin == pytest.approx(_margin_reference(offset, dz), rel=1e-13, abs=0.0)
