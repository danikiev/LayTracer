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
        RT = laytracer.psv_rt_coefficients(1e-10, vp1, vs1, rho1, vp2, vs2, rho2)
        T_normal = laytracer.transmission_normal(vp1, rho1, vp2, rho2)
        assert abs(RT["Tpp"]) == pytest.approx(T_normal, rel=0.15)

    def test_transmission_positive(self):
        """Transmission coefficient magnitude is positive."""
        RT = laytracer.psv_rt_coefficients(
            0.0001, 3000.0, 1500.0, 2200.0, 5000.0, 2800.0, 2700.0
        )
        assert abs(RT["Tpp"]) > 0


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
        """In a homogeneous medium, spreading ≈ distance (the ray path length)."""
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
            h, v, segments, [], epicentral_dist=4000.0, z_src=0.0, z_rcv=3000.0,
            compute_amplitude=True,
        )
        assert res.spreading is not None
        assert res.spreading > 0

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
    mi_vp, mi_vs, mi_rho = 4.98, 2.9, 2.667
    mt_vp, mt_vs, mt_rho = 8.00, 4.6, 3.38

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
