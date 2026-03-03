r"""Tests for amplitude computation: t*, spreading, and transmission coefficients."""

import numpy as np
import pandas as pd
import pytest

import laytracer
from laytracer.solver import offset, offset_dq, offset_dq2, q_from_p


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
#  Theory Verification
# ═══════════════════════════════════════════════════════════════════════

class TestTheoryVerification:
    """Rigorous quantitative verification of mathematical theory for t* and spreading."""
    
    def test_tstar_theoretical_formula(self):
        """t* must exactly match the mathematical sum of layer-by-layer attenuation dt / Q."""
        h = np.array([500.0, 1000.0, 1500.0])
        v = np.array([3000.0, 4500.0, 6000.0])
        Q = np.array([100.0, 200.0, 300.0])
        
        # Pick an arbitrary valid ray parameter (must be < 1/6000)
        p = 0.0001
        
        # 1. Theoretical calculation exactly per formula
        eta = np.sqrt(1.0 / v**2 - p**2)
        dt = h / (v**2 * eta)
        tstar_theory = float(np.sum(dt / Q))
        
        # 2. Recreate condition for the solver
        X_target = float(np.sum(h * p / eta))
        
        segments = [{
            "h": h, "v": v, "vp": v, "vs": v/1.732, "rho": v*0.5+1000.0,
            "qp": Q, "qs": Q/2.0, "phase": "P", 
            "start_z": 0.0, "end_z": 3000.0
        }]
        
        # Run solver exactly to the target X
        res = laytracer.solve(
            h, v, segments, [], epicentral_dist=X_target, z_src=0.0, z_rcv=3000.0,
            compute_amplitude=True, tol=1e-10
        )
        
        # Verify it matched the expected parameter and the theoretical tstar formula exactly
        assert res.ray_parameter == pytest.approx(p, rel=1e-5)
        assert res.tstar == pytest.approx(tstar_theory, rel=1e-6)

    def test_spreading_analytical_derivative(self):
        """Spreading relies on analytical dX/dp. Verify perfectly matches finite difference."""
        h = np.array([500.0, 1000.0, 1500.0])
        v = np.array([3000.0, 4500.0, 6000.0])
        
        # Arbitrary valid p
        p = 0.0001
        
        # Definition of X(p)
        def X_of_p(p_val):
            eta = np.sqrt(1.0 / v**2 - p_val**2)
            return np.sum(h * p_val / eta)
            
        X_target = X_of_p(p)
        
        # 1. Central finite difference for true dX/dp
        eps = 1e-8
        dXdp_fd = (X_of_p(p + eps) - X_of_p(p - eps)) / (2 * eps)
        
        # 2. Analytical dX/dp using internal q-derivatives
        vmax = float(np.max(v))
        lmd = v / vmax
        q = p * vmax / np.sqrt(1.0 - (p * vmax)**2)
        
        dXdq = offset_dq(q, h, lmd)
        dqdp = vmax / (1.0 - (p * vmax)**2)**1.5
        dXdp_analytic = dXdq * dqdp
        
        # Prove the code's analytical derivatives precisely match numerical math
        assert dXdp_analytic == pytest.approx(dXdp_fd, rel=1e-5)
        
        # 3. Verify the final spreading factor formula computes correctly
        # The current implementation produces spreading = distance * velocity in homogeneous media.
        cos_is = np.sqrt(1.0 - (p * v[0])**2)
        cos_ir = np.sqrt(1.0 - (p * v[-1])**2)
        L_theory = np.sqrt(X_target * abs(dXdp_analytic) * cos_is * cos_ir / max(p, 1e-15))
        
        segments = [{
            "h": h, "v": v, "vp": v, "vs": v/1.732, "rho": v*0.5+1000.0,
            "qp": v*0 + 500.0, "qs": v*0 + 250.0, "phase": "P", 
            "start_z": 0.0, "end_z": 3000.0
        }]
        
        # Run solver exactly to the target X
        res = laytracer.solve(
            h, v, segments, [], epicentral_dist=X_target, z_src=0.0, z_rcv=3000.0,
            compute_amplitude=True, tol=1e-10
        )
        
        assert res.ray_parameter == pytest.approx(p, rel=1e-5)
        assert res.spreading == pytest.approx(L_theory, rel=1e-6)

    def test_offset_residual_matches_target(self):
        """After solving, verify the returned ray parameter reproduces the target offset."""
        h = np.array([500.0, 1000.0, 1500.0])
        v = np.array([3000.0, 4500.0, 6000.0])
        X_target = 7000.0
        
        vmax = float(v.max())
        lmd = v / vmax
        
        segments = [{
            "h": h, "v": v, "vp": v, "vs": v/1.732, "rho": np.ones_like(v)*2500.,
            "qp": np.ones_like(v)*300., "qs": np.ones_like(v)*150., "phase": "P",
            "start_z": 0.0, "end_z": 3000.0
        }]
        
        res = laytracer.solve(
            h, v, segments, [], X_target, 0.0, 3000.0,
            compute_amplitude=False, tol=1e-10
        )
        
        # Recompute offset from returned p using internal q logic
        q_back = q_from_p(res.ray_parameter, vmax)
        X_back = offset(q_back, h, lmd)
        assert X_back == pytest.approx(X_target, rel=1e-10, abs=1e-6)

    def test_offset_derivatives_fd(self):
        """Direct finite-difference test for X'(q) and X''(q)."""
        rng = np.random.default_rng(0)
        h = rng.uniform(100, 2000, size=5)
        lmd = rng.uniform(0.4, 1.0, size=5)
        q = 2.3
        eps = 1e-4  # larger step for more stable second derivative FD
        
        Xp = offset(q + eps, h, lmd)
        Xm = offset(q - eps, h, lmd)
        X0 = offset(q, h, lmd)
        
        d1_fd = (Xp - Xm) / (2 * eps)
        d2_fd = (Xp - 2*X0 + Xm) / (eps**2)
        
        assert offset_dq(q, h, lmd) == pytest.approx(d1_fd, rel=1e-5)
        # Second derivative FD is less accurate; loosen tolerance slightly
        assert offset_dq2(q, h, lmd) == pytest.approx(d2_fd, rel=1e-2)


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
        """Angle-dependent RT coefficients at normal incidence (p=0) match simple theory."""
        vp1, vs1, rho1 = 4000.0, 2000.0, 2500.0
        vp2, vs2, rho2 = 5000.0, 2800.0, 2700.0
        
        # At exactly p=0, Zoeppritz reduces to simple impedance ratios
        RT = laytracer.psv_rt_coefficients(0.0, vp1, vs1, rho1, vp2, vs2, rho2)
        
        # 1. Tpp must match the P-wave impedance-based formula
        T_normal_p = laytracer.transmission_normal(vp1, rho1, vp2, rho2)
        assert abs(RT["Tpp"]) == pytest.approx(T_normal_p, rel=1e-10)

        # 2. Tss must match the S-wave impedance-based formula
        T_normal_s = laytracer.transmission_normal(vs1, rho1, vs2, rho2)
        assert abs(RT["Tss"]) == pytest.approx(T_normal_s, rel=1e-10)

        # 3. Converted modes (P<->SV) must vanish exactly (at numerical precision)
        for key in ["Rps", "Tps", "Rsp", "Tsp"]:
            assert abs(RT[key]) == pytest.approx(0.0, abs=1e-15)

    def test_transmission_positive(self):
        """Transmission coefficient magnitude is positive."""
        RT = laytracer.psv_rt_coefficients(
            0.0001, 3000.0, 1500.0, 2200.0, 5000.0, 2800.0, 2700.0
        )
        assert abs(RT["Tpp"]) > 0

    def test_zoeppritz_converted_modes_vanish_at_normal_incidence(self):
        """At p=0, Rps, Tps, Rsp, Tsp must be exactly zero."""
        vp1, vs1, rho1 = 4000.0, 2000.0, 2500.0
        vp2, vs2, rho2 = 5000.0, 2800.0, 2700.0
        RTp = laytracer.psv_rt_coefficients(0.0, vp1, vs1, rho1, vp2, vs2, rho2)
        assert abs(RTp["Rps"]) == pytest.approx(0.0, abs=1e-8)
        assert abs(RTp["Tps"]) == pytest.approx(0.0, abs=1e-8)
        assert abs(RTp["Rsp"]) == pytest.approx(0.0, abs=1e-8)
        assert abs(RTp["Tsp"]) == pytest.approx(0.0, abs=1e-8)


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
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": 0.0, "end_z": 3000.0
        }]
        
        res = laytracer.solve(
            h, v, segments, [], epicentral_dist=0.0, z_src=0.0, z_rcv=3000.0,
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
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": 100.0, "end_z": 2500.0
        }]
        
        res = laytracer.solve(
            h, v, segments, [], epicentral_dist=5000.0, z_src=100.0, z_rcv=2500.0,
            compute_amplitude=True,
        )
        assert res.tstar == pytest.approx(res.travel_time / Q, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════
#  Geometrical spreading
# ═══════════════════════════════════════════════════════════════════════

class TestSpreading:
    def test_spreading_homogeneous(self):
        """In a homogeneous medium, relative geometrical spreading = distance (ray path length) * velocity."""
        df = pd.DataFrame({
            "Depth": [0.0], "Vp": [5000.0], "Vs": [2887.0],
            "Rho": [2700.0], "Qp": [500.0], "Qs": [250.0],
        })
        stack = laytracer.build_layer_stack(df, z_src=0.0, z_rcv=3000.0)
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": 0.0, "end_z": 3000.0
        }]
        
        X = 4000.0
        z_src, z_rcv = 0.0, 3000.0
        res = laytracer.solve(
            h, v, segments, [], epicentral_dist=X, z_src=z_src, z_rcv=z_rcv,
            compute_amplitude=True,
        )
        dist = np.sqrt(X**2 + (z_rcv - z_src)**2)
        # Definition: relative spreading = distance * velocity (formally at receiver)
        assert res.spreading == pytest.approx(dist * v[-1], rel=1e-5)

    def test_spreading_equals_distance_in_homogeneous_multilayer(self):
        """Force the solver through multi-layer logic and verify relative geometrical spreading still reduces to distance * velocity."""
        # Build 3 identical layers so it uses general logic (N > 1)
        h = np.array([1000.0, 1000.0, 1000.0])
        v = np.array([5000.0, 5000.0, 5000.0])
        X = 4000.0
        z_src, z_rcv = 0.0, 3000.0

        segments = [{
            "h": h, "v": v, "vp": v, "vs": v/1.732, "rho": np.ones_like(v)*2700.0,
            "qp": np.ones_like(v)*500.0, "qs": np.ones_like(v)*250.0, "phase": "P",
            "start_z": z_src, "end_z": z_rcv,
        }]

        res = laytracer.solve(
            h, v, segments, [], X, z_src, z_rcv, 
            compute_amplitude=True, tol=1e-10
        )

        dist = np.sqrt(X**2 + (z_rcv - z_src)**2)
        # Formally use v[-1] at receiver, but all layers have same velocity
        assert res.spreading == pytest.approx(dist * v[-1], rel=1e-5)

    def test_spreading_positive_multilayer(self):
        """Spreading is positive for a multi-layer model."""
        df = _simple_model()
        stack = laytracer.build_layer_stack(df, z_src=500.0, z_rcv=2500.0)
        
        h = stack.h
        v = stack.vp
        segments = [{
            "h": h, "v": v, "vp": stack.vp, "vs": stack.vs, "rho": stack.rho,
            "qp": stack.qp, "qs": stack.qs, "phase": "P", 
            "start_z": 500.0, "end_z": 2500.0
        }]
        
        res = laytracer.solve(
            h, v, segments, [], epicentral_dist=5000.0, z_src=500.0, z_rcv=2500.0,
            compute_amplitude=True,
        )
        assert res.spreading is not None
        assert res.spreading > 0

# ═══════════════════════════════════════════════════════════════════════
#  Brewster-angle detection
# ═══════════════════════════════════════════════════════════════════════

def _ammon_model_rt(n=1000):
    """Compute RT coefficients for Ammon's crust/mantle test case.

    Returns (RT_P, angle_P, RT_SV, angle_SV).
    """
    # Parameters in m/s and kg/m^3
    mi_vp, mi_vs, mi_rho = 4.98e3, 2.9e3, 2667.0
    mt_vp, mt_vs, mt_rho = 8.00e3, 4.6e3, 3380.0

    # Ray parameters in s/m
    p_P = np.linspace(0, 1.0 / mi_vp, n + 1)
    p_SV = np.linspace(0, 1.0 / mi_vs, n + 1)

    RT_P = laytracer.psv_rt_coefficients(
        p_P, mi_vp, mi_vs, mi_rho, mt_vp, mt_vs, mt_rho,
    )
    RT_SV = laytracer.psv_rt_coefficients(
        p_SV, mi_vp, mi_vs, mi_rho, mt_vp, mt_vs, mt_rho,
    )
    angle_P = np.rad2deg(np.arcsin(np.clip(p_P * mi_vp, -1, 1)))
    angle_SV = np.rad2deg(np.arcsin(np.clip(p_SV * mi_vs, -1, 1)))
    return RT_P, angle_P, RT_SV, angle_SV


class TestBrewsterAngles:
    """Tests for :func:`laytracer.find_brewster_angles`."""

    def test_no_brewster_for_identical_media(self):
        """Identical half-spaces produce no Brewster angles."""
        n = 500
        p = np.linspace(0, 1.0 / 5000.0, n + 1)
        RT = laytracer.psv_rt_coefficients(
            p, 5000.0, 2887.0, 2700.0, 5000.0, 2887.0, 2700.0,
        )
        angles = np.rad2deg(np.arcsin(np.clip(p * 5000.0, -1, 1)))
        result = laytracer.find_brewster_angles(RT, angles)
        assert result == {}

    def test_p_incident_rps_brewster(self):
        """For Ammon model, |Rps| has a Brewster angle near 37.9°."""
        RT_P, angle_P, _, _ = _ammon_model_rt()
        result = laytracer.find_brewster_angles(RT_P, angle_P, keys=["Rps"])
        assert "Rps" in result
        assert len(result["Rps"]) == 1
        assert result["Rps"][0] == pytest.approx(37.9, abs=0.5)

    def test_sv_incident_rss_brewster(self):
        """For Ammon model, |Rss| has a Brewster angle near 19.8°."""
        _, _, RT_SV, angle_SV = _ammon_model_rt()
        result = laytracer.find_brewster_angles(RT_SV, angle_SV, keys=["Rss"])
        assert "Rss" in result
        assert len(result["Rss"]) == 1
        assert result["Rss"][0] == pytest.approx(19.8, abs=0.5)

    def test_sv_incident_rsp_two_brewsters(self):
        """For Ammon model, |Rsp| has two Brewster angles near 21° and 40°."""
        _, _, RT_SV, angle_SV = _ammon_model_rt()
        result = laytracer.find_brewster_angles(RT_SV, angle_SV, keys=["Rsp"])
        assert "Rsp" in result
        assert len(result["Rsp"]) == 2
        assert result["Rsp"][0] == pytest.approx(21.0, abs=1.0)
        assert result["Rsp"][1] == pytest.approx(40.0, abs=1.0)

    def test_keys_filter(self):
        """Only requested keys appear in the output."""
        RT_P, angle_P, _, _ = _ammon_model_rt()
        result = laytracer.find_brewster_angles(RT_P, angle_P, keys=["Tpp"])
        # Tpp has no Brewster minimum below 0.05, so it should be absent
        assert "Tpp" not in result
        assert "Rps" not in result  # not requested

    def test_threshold_controls_sensitivity(self):
        """Raising the threshold may detect more minima."""
        RT_P, angle_P, _, _ = _ammon_model_rt()
        strict = laytracer.find_brewster_angles(
            RT_P, angle_P, keys=["Rps"], threshold=0.001,
        )
        loose = laytracer.find_brewster_angles(
            RT_P, angle_P, keys=["Rps"], threshold=0.1,
        )
        # Loose threshold should find at least as many as strict
        n_strict = len(strict.get("Rps", []))
        n_loose = len(loose.get("Rps", []))
        assert n_loose >= n_strict
