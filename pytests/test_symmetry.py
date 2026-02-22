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
        """t*(S,R) = t*(R,S). Path integral must not depend on direction."""
        df = pd.DataFrame({
            "Depth": [0.0, 1000.0, 2000.0, 3500.0],
            "Vp":    [3000.0, 4500.0, 5500.0, 6500.0],
            "Vs":    [1500.0, 2250.0, 2750.0, 3250.0],
            "Rho":   [2200.0, 2500.0, 2700.0, 2900.0],
            "Qp":    [200.0,  50.0,   600.0,  800.0],
            "Qs":    [100.0,  25.0,   300.0,  400.0],
        })
        src = np.array([0.0, 0.0, 3000.0])
        rcv = np.array([8000.0, 0.0, 0.0])
        r_fwd = laytracer.trace_rays(src, rcv, df, compute_amplitude=True)
        r_rev = laytracer.trace_rays(rcv, src, df, compute_amplitude=True)
        assert r_fwd.tstar[0] == pytest.approx(r_rev.tstar[0], rel=1e-6)

    def test_spreading_reciprocity(self):
        """L(S,R) = L(R,S) (Červený-style invariant)."""
        # Spreading is invariant under source-receiver swap
        df = pd.DataFrame({
            "Depth": [0.0, 1000.0, 2000.0, 3500.0],
            "Vp":    [3000.0, 4500.0, 5500.0, 6500.0],
            "Vs":    [1500.0, 2250.0, 2750.0, 3250.0],
            "Rho":   [2200.0, 2500.0, 2700.0, 2900.0],
            "Qp":    [200.0,  50.0,   600.0,  800.0],
            "Qs":    [100.0,  25.0,   300.0,  400.0],
        })
        src = np.array([0.0, 0.0, 3000.0])
        rcv = np.array([8000.0, 0.0, 0.0])
        r_fwd = laytracer.trace_rays(src, rcv, df, compute_amplitude=True)
        r_rev = laytracer.trace_rays(rcv, src, df, compute_amplitude=True)
        assert r_fwd.spreading[0] == pytest.approx(r_rev.spreading[0], rel=1e-5)

    def test_ray_path_reversal(self):
        """Forward ray ≈ flipped reverse ray (geometry invariance)."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        r_fwd = laytracer.trace_rays(src, rcv, df)
        r_rev = laytracer.trace_rays(rcv, src, df)

        path1 = r_fwd.rays[0]
        path2 = r_rev.rays[0]

        # Calculate horizontal range from source for each point
        range1 = np.sqrt(np.sum((path1[:, :2] - src[:2])**2, axis=1))
        # path2 starts at RCV and ends at SRC. Reversed, it starts at SRC and ends at RCV.
        # This makes its range-from-SRC directly comparable to range1.
        range2_rev = np.sqrt(np.sum((path2[::-1, :2] - src[:2])**2, axis=1))
        z1 = path1[:, 2]
        z2_rev = path2[::-1, 2]

        # Robust check: compare ranges only at corresponding depths.
        # This handles cases where sampling might differ but the path curve is the same.
        common_z = np.intersect1d(z1, z2_rev)
        assert len(common_z) >= 2  # Must at least have src/rcv depths
        
        for z in common_z:
            # Get range at this depth for both rays
            r1 = range1[np.argmin(np.abs(z1 - z))]
            r2 = range2_rev[np.argmin(np.abs(z2_rev - z))]
            assert r1 == pytest.approx(r2, abs=1e-3)

    def test_transmission_product_ratio(self):
        """Displacement transmission products are reciprocal up to an impedance ratio."""
        # For Aki & Richards displacement coefficients:
        # T_fwd / T_rev = (rho_s * v_s * cos_s) / (rho_r * v_r * cos_r)
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])

        fwd = laytracer.trace_rays(src, rcv, df, compute_amplitude=True)
        rev = laytracer.trace_rays(rcv, src, df, compute_amplitude=True)

        T_fwd = fwd.trans_product[0]
        T_rev = rev.trans_product[0]

        # Get velocities and densities at source and receiver
        # src at 500m -> Layer 0
        # rcv at 2500m -> Layer 2
        v_s, rho_s = 3000.0, 2200.0
        v_r, rho_r = 6000.0, 2800.0

        p = fwd.ray_parameters[0]
        cos_s = np.sqrt(1.0 - (p * v_s)**2)
        cos_r = np.sqrt(1.0 - (p * v_r)**2)

        expected_ratio = (rho_s * v_s * cos_s) / (rho_r * v_r * cos_r)
        actual_ratio = T_fwd / T_rev

        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-4)

    def test_traveltime_independent_of_azimuth(self):
        """Travel time and ray parameter don't depend on horizontal direction."""
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

        # Travel times
        assert r1.travel_times[0] == pytest.approx(r2.travel_times[0], rel=1e-6)
        assert r1.travel_times[0] == pytest.approx(r3.travel_times[0], rel=1e-6)

        # Ray parameters
        assert r1.ray_parameters[0] == pytest.approx(r2.ray_parameters[0], rel=1e-6)
        assert r1.ray_parameters[0] == pytest.approx(r3.ray_parameters[0], rel=1e-6)

    def test_translation_invariance(self):
        """In 1D models, absolute x/y doesn't matter—only relative offset."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        shift = np.array([1234.0, -987.0, 0.0])

        r0 = laytracer.trace_rays(src, rcv, df, compute_amplitude=True)
        rS = laytracer.trace_rays(src + shift, rcv + shift, df, compute_amplitude=True)

        assert r0.travel_times[0] == pytest.approx(rS.travel_times[0], rel=1e-6)
        assert r0.ray_parameters[0] == pytest.approx(rS.ray_parameters[0], rel=1e-6)
        assert r0.spreading[0] == pytest.approx(rS.spreading[0], rel=1e-6)
        assert r0.tstar[0] == pytest.approx(rS.tstar[0], rel=1e-6)

    def test_path_shape_azimuth_invariance(self):
        """Ray paths at different azimuths have same (range, depth) shape."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        # Same distance, different directions
        rcv1 = np.array([5000.0, 0.0, 2500.0])
        a = 5000.0 / np.sqrt(2)
        rcv2 = np.array([a, a, 2500.0])

        r1 = laytracer.trace_rays(src, rcv1, df)
        r2 = laytracer.trace_rays(src, rcv2, df)

        path1 = r1.rays[0]
        path2 = r2.rays[0]

        # Calculate horizontal range from source for each point
        range1 = np.sqrt(np.sum((path1[:, :2] - src[:2])**2, axis=1))
        range2 = np.sqrt(np.sum((path2[:, :2] - src[:2])**2, axis=1))

        # Range and depth should match perfectly
        np.testing.assert_allclose(range1, range2, atol=1e-5)
        np.testing.assert_allclose(path1[:, 2], path2[:, 2], atol=1e-5)
