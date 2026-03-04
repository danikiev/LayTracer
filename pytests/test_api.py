r"""Tests for the high-level trace_rays() API."""

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


class TestTraceRays:
    def test_single_pair(self):
        """One source → one receiver."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        result = laytracer.trace_rays(src, rcv, df)

        assert result.travel_times.shape == (1,)
        assert result.travel_times[0] > 0
        assert len(result.rays) == 1
        assert result.rays[0].shape[1] == 3

    def test_multiple_receivers(self):
        """One source → multiple receivers."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcvs = np.array([
            [3000.0, 0.0, 0.0],
            [5000.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0],
        ])
        result = laytracer.trace_rays(src, rcvs, df)
        assert result.travel_times.shape == (3,)
        assert len(result.rays) == 3

    def test_multiple_sources(self):
        """Multiple sources × multiple receivers."""
        df = _simple_model()
        srcs = np.array([[0.0, 0.0, 2500.0], [1000.0, 0.0, 1500.0]])
        rcvs = np.array([[3000.0, 0.0, 0.0], [6000.0, 0.0, 0.0]])
        result = laytracer.trace_rays(srcs, rcvs, df)
        # 2 sources × 2 receivers = 4 rays
        assert result.travel_times.shape == (4,)
        assert len(result.rays) == 4

    def test_parallel_matches_sequential(self):
        """Parallel and sequential produce the same results."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcvs = np.array([
            [3000.0, 0.0, 0.0],
            [5000.0, 1000.0, 0.0],
            [7000.0, -2000.0, 0.0],
        ])
        r_seq = laytracer.trace_rays(src, rcvs, df, n_jobs=1)
        r_par = laytracer.trace_rays(src, rcvs, df, n_jobs=2, sequential_limit=0)

        np.testing.assert_allclose(
            r_seq.travel_times, r_par.travel_times, rtol=1e-10
        )

    def test_with_amplitude(self):
        """Amplitude computation returns t*, spreading, trans_product."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        result = laytracer.trace_rays(
            src, rcv, df, compute_amplitude=True
        )
        assert result.tstar is not None
        assert result.spreading is not None
        assert result.trans_product is not None

    def test_same_depth_horizontal_ray(self):
        """Source and receiver at the same depth must produce valid results."""
        df = _simple_model()
        # Station at z=0, grid point at z=0 with horizontal offset
        src = np.array([0.0, 0.0, 0.0])
        rcv = np.array([5000.0, 0.0, 0.0])
        result = laytracer.trace_rays(
            src, rcv, df, compute_amplitude=True
        )
        # Travel time must be finite and positive
        assert np.isfinite(result.travel_times[0])
        assert result.travel_times[0] > 0
        # Expected: epic / Vp(layer 0)
        expected_tt = 5000.0 / 3000.0
        assert result.travel_times[0] == pytest.approx(expected_tt, rel=1e-6)
        # Amplitude quantities must be finite
        assert np.isfinite(result.tstar[0])
        assert np.isfinite(result.spreading[0])
        assert np.isfinite(result.trans_product[0])
        # Spreading = epic * v for homogeneous medium
        assert result.spreading[0] == pytest.approx(5000.0 * 3000.0, rel=1e-6)
        # No interface crossed → T = 1
        assert result.trans_product[0] == pytest.approx(1.0)

    def test_same_depth_no_amplitude(self):
        """Same-depth ray without amplitude computation."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 1500.0])
        rcv = np.array([3000.0, 4000.0, 1500.0])
        result = laytracer.trace_rays(src, rcv, df, compute_amplitude=False)
        epic = 5000.0
        expected_tt = epic / 4500.0  # layer 1 velocity
        assert result.travel_times[0] == pytest.approx(expected_tt, rel=1e-6)
        assert result.tstar is None
        assert result.spreading is None

    def test_same_point_degenerate(self):
        """Source and receiver at the exact same point."""
        df = _simple_model()
        src = np.array([100.0, 200.0, 500.0])
        rcv = np.array([100.0, 200.0, 500.0])
        result = laytracer.trace_rays(
            src, rcv, df, compute_amplitude=True
        )
        assert result.travel_times[0] == pytest.approx(0.0)

    def test_3d_ray_path_coords(self):
        """3-D ray path starts at source and ends at receiver."""
        df = _simple_model()
        src = np.array([1000.0, 2000.0, 500.0])
        rcv = np.array([4000.0, 5000.0, 2500.0])
        result = laytracer.trace_rays(src, rcv, df)
        ray = result.rays[0]
        # Start point
        np.testing.assert_allclose(ray[0], src, atol=1.0)
        # End point z
        assert ray[-1, 2] == pytest.approx(rcv[2], abs=1.0)
