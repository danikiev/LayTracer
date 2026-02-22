r"""Tests for reciprocity / symmetry of ray tracing results."""

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


class TestSymmetry:
    def test_traveltime_reciprocity(self):
        """tt(A→B) = tt(B→A)."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        r_fwd = laytracer.trace_rays(src, rcv, df)
        r_rev = laytracer.trace_rays(rcv, src, df)
        assert r_fwd.travel_times[0] == pytest.approx(
            r_rev.travel_times[0], rel=1e-6
        )

    def test_ray_parameter_reciprocity(self):
        """Same ray parameter in both directions."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        r_fwd = laytracer.trace_rays(src, rcv, df)
        r_rev = laytracer.trace_rays(rcv, src, df)
        assert r_fwd.ray_parameters[0] == pytest.approx(
            r_rev.ray_parameters[0], rel=1e-6
        )

    def test_tstar_reciprocity(self):
        """t* is the same from A→B and B→A."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        r_fwd = laytracer.trace_rays(src, rcv, df, compute_amplitude=True)
        r_rev = laytracer.trace_rays(rcv, src, df, compute_amplitude=True)
        assert r_fwd.tstar[0] == pytest.approx(r_rev.tstar[0], rel=1e-4)

    def test_ray_path_reversal(self):
        """Forward ray ≈ flipped reverse ray (z-coordinates)."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        r_fwd = laytracer.trace_rays(src, rcv, df)
        r_rev = laytracer.trace_rays(rcv, src, df)
        z_fwd = r_fwd.rays[0][:, 2]
        z_rev = r_rev.rays[0][::-1, 2]
        np.testing.assert_allclose(z_fwd, z_rev, atol=1.0)

    def test_traveltime_independent_of_azimuth(self):
        """Travel time doesn't depend on horizontal direction."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        # Same epicentral distance, different azimuths
        rcv1 = np.array([5000.0, 0.0, 2500.0])
        rcv2 = np.array([0.0, 5000.0, 2500.0])
        a = 5000.0 / np.sqrt(2)
        rcv3 = np.array([a, a, 2500.0])
        r1 = laytracer.trace_rays(src, rcv1, df)
        r2 = laytracer.trace_rays(src, rcv2, df)
        r3 = laytracer.trace_rays(src, rcv3, df)
        assert r1.travel_times[0] == pytest.approx(r2.travel_times[0], rel=1e-6)
        assert r1.travel_times[0] == pytest.approx(r3.travel_times[0], rel=1e-6)
