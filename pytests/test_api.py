r"""Tests for the high-level trace_rays() API."""

import inspect
import pickle
import numpy as np
import pandas as pd
import psutil
import pytest
from dataclasses import fields
from types import SimpleNamespace

import laytracer
from laytracer.api import TRACE_OUTPUTS, _trace_batch, _unpack_results
from laytracer.model import ModelArrays


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
        """One source ??? one receiver."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        result = laytracer.trace_rays(src, rcv, df)

        assert result.travel_times.shape == (1,)
        assert result.travel_times[0] > 0
        assert len(result.rays) == 1
        assert result.rays[0].shape[1] == 3
        assert result.source_phase == "P"

    def test_s_alias_matches_sv(self):
        """Legacy S phase is treated as SV."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        requested = {"travel_times", "rays", "ray_parameters", "tstar", "spreading", "trans_product"}

        result_s = laytracer.trace_rays(src, rcv, df, source_phase="S", requested=requested)
        result_sv = laytracer.trace_rays(src, rcv, df, source_phase="SV", requested=requested)

        assert result_s.source_phase == "SV"
        np.testing.assert_allclose(result_s.travel_times, result_sv.travel_times)
        np.testing.assert_allclose(result_s.ray_parameters, result_sv.ray_parameters)
        np.testing.assert_allclose(result_s.tstar, result_sv.tstar)
        np.testing.assert_allclose(result_s.spreading, result_sv.spreading)
        np.testing.assert_allclose(result_s.trans_product, result_sv.trans_product)

    def test_multi_phase_returns_phase_results(self):
        """A phase list returns one TraceResult per canonical source phase."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        requested = {"travel_times", "rays", "ray_parameters", "tstar", "spreading", "trans_product"}

        multi = laytracer.trace_rays(
            src,
            rcv,
            df,
            source_phase=["P", "SH", "SV", "S"],
            requested=requested,
        )

        assert set(multi) == {"P", "SH", "SV"}
        assert multi["P"].source_phase == "P"
        assert multi["SH"].source_phase == "SH"
        assert multi["SV"].source_phase == "SV"

        single_sv = laytracer.trace_rays(
            src, rcv, df, source_phase="SV", requested=requested
        )
        np.testing.assert_allclose(multi["SV"].travel_times, single_sv.travel_times)
        np.testing.assert_allclose(multi["SV"].ray_parameters, single_sv.ray_parameters)
        np.testing.assert_allclose(multi["SV"].trans_product, single_sv.trans_product)

    def test_sv_and_sh_share_kinematics_but_not_coefficients(self):
        """SV and SH share Vs kinematics but use different coefficient families."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        requested = {"travel_times", "rays", "ray_parameters", "tstar", "spreading", "trans_product"}

        result = laytracer.trace_rays(
            src, rcv, df, source_phase=["SV", "SH"], requested=requested
        )

        np.testing.assert_allclose(result["SV"].travel_times, result["SH"].travel_times)
        np.testing.assert_allclose(result["SV"].ray_parameters, result["SH"].ray_parameters)
        np.testing.assert_allclose(result["SV"].tstar, result["SH"].tstar)
        np.testing.assert_allclose(result["SV"].spreading, result["SH"].spreading)
        np.testing.assert_allclose(result["SV"].rays[0], result["SH"].rays[0])
        assert not np.isclose(result["SV"].trans_product[0], result["SH"].trans_product[0])

    def test_upward_sv_and_sh_share_tstar_and_spreading(self):
        """Upward SV and SH direct rays share Vs/Qs attenuation and spreading."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 2500.0])
        rcv = np.array([5000.0, 0.0, 500.0])
        requested = {"travel_times", "rays", "ray_parameters", "tstar", "spreading"}

        result = laytracer.trace_rays(
            src, rcv, df, source_phase=["SV", "SH"], requested=requested
        )

        np.testing.assert_allclose(result["SV"].travel_times, result["SH"].travel_times)
        np.testing.assert_allclose(result["SV"].ray_parameters, result["SH"].ray_parameters)
        np.testing.assert_allclose(result["SV"].tstar, result["SH"].tstar)
        np.testing.assert_allclose(result["SV"].spreading, result["SH"].spreading)
        np.testing.assert_allclose(result["SV"].rays[0], result["SH"].rays[0])

    def test_multi_phase_duplicate_alias_returns_one_entry(self):
        """Duplicate canonical phases are de-duplicated."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        result = laytracer.trace_rays(src, rcv, df, source_phase=["S", "SV"])

        assert set(result) == {"SV"}
        assert result["SV"].source_phase == "SV"

    def test_invalid_source_phase(self):
        """Invalid source phases fail explicitly."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        with pytest.raises(ValueError, match="Invalid phase"):
            laytracer.trace_rays(src, rcv, df, source_phase="X")

    def test_multiple_receivers(self):
        """One source ??? multiple receivers."""
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
        """Multiple sources ?? multiple receivers."""
        df = _simple_model()
        srcs = np.array([[0.0, 0.0, 2500.0], [1000.0, 0.0, 1500.0]])
        rcvs = np.array([[3000.0, 0.0, 0.0], [6000.0, 0.0, 0.0]])
        result = laytracer.trace_rays(srcs, rcvs, df)
        # 2 sources ?? 2 receivers = 4 rays
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
        try:
            r_par = laytracer.trace_rays(
                src, rcvs, df, n_jobs=2, backend="threading", sequential_limit=0
            )
        except PermissionError:
            pytest.skip("Parallel worker creation is blocked in this environment")

        np.testing.assert_allclose(
            r_seq.travel_times, r_par.travel_times, rtol=1e-10
        )

    def test_with_amplitude(self):
        """Amplitude computation returns t*, spreading, trans_product."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([5000.0, 0.0, 2500.0])
        result = laytracer.trace_rays(
            src, rcv, df, requested={"travel_times", "rays", "ray_parameters", "tstar", "spreading", "trans_product"}
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
            src, rcv, df, requested={"travel_times", "rays", "ray_parameters", "tstar", "spreading", "trans_product"}
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
        # No interface crossed ??? T = 1
        assert result.trans_product[0] == pytest.approx(1.0)

    def test_same_depth_no_amplitude(self):
        """Same-depth ray without amplitude computation."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 1500.0])
        rcv = np.array([3000.0, 4000.0, 1500.0])
        result = laytracer.trace_rays(src, rcv, df, requested={"travel_times", "rays", "ray_parameters"})
        epic = 5000.0
        expected_tt = epic / 4500.0  # layer 1 velocity
        assert result.travel_times[0] == pytest.approx(expected_tt, rel=1e-6)
        assert result.tstar is None
        assert result.spreading is None

    def test_zero_offset_vertical_ray_has_finite_amplitudes(self):
        """Zero-offset direct rays through multiple layers must keep finite amplitudes."""
        df = _simple_model()
        src = np.array([0.0, 0.0, 500.0])
        rcv = np.array([0.0, 0.0, 1500.0])
        result = laytracer.trace_rays(
            src, rcv, df, requested={"travel_times", "rays", "ray_parameters", "tstar", "spreading", "trans_product"}, n_jobs=1
        )

        expected_tt = 500.0 / 3000.0 + 500.0 / 4500.0
        expected_tstar = 500.0 / (3000.0 * 200.0) + 500.0 / (4500.0 * 400.0)
        expected_spreading = 500.0 * 3000.0 + 500.0 * 4500.0
        expected_trans = abs(
            laytracer.transmission_normal(3000.0, 2200.0, 4500.0, 2500.0)
        )

        assert result.travel_times[0] == pytest.approx(expected_tt, rel=1e-6)
        assert result.tstar[0] == pytest.approx(expected_tstar, rel=1e-6)
        assert result.spreading[0] == pytest.approx(expected_spreading, rel=1e-6)
        assert result.trans_product[0] == pytest.approx(expected_trans, rel=1e-6)
        assert np.isfinite(result.tstar[0])
        assert np.isfinite(result.spreading[0])
        assert np.isfinite(result.trans_product[0])

    def test_same_point_degenerate(self):
        """Source and receiver at the exact same point."""
        df = _simple_model()
        src = np.array([100.0, 200.0, 500.0])
        rcv = np.array([100.0, 200.0, 500.0])
        result = laytracer.trace_rays(
            src, rcv, df, requested={"travel_times", "rays", "ray_parameters", "tstar", "spreading", "trans_product"}
        )
        assert result.travel_times[0] == pytest.approx(0.0)
        assert result.tstar[0] == pytest.approx(0.0)
        assert result.spreading[0] == pytest.approx(0.0)
        assert result.trans_product[0] == pytest.approx(1.0)
        assert np.isfinite(result.spreading[0])

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


def test_unpack_results_handles_mixed_none_amplitudes():
    """Mixed finite/None amplitude values should unpack to arrays with NaNs."""
    results = [
        (1.0, np.array([[0.0, 0.0, 0.0]]), 0.1, 0.2, None, 1.0),
        (2.0, np.array([[1.0, 0.0, 0.0]]), 0.2, 0.3, 4.5, None),
    ]

    unpacked = _unpack_results(results, requested={"travel_times", "rays", "ray_parameters", "tstar", "spreading", "trans_product"})

    np.testing.assert_allclose(unpacked.travel_times, [1.0, 2.0])
    np.testing.assert_allclose(unpacked.tstar, [0.2, 0.3])
    assert np.isnan(unpacked.spreading[0])
    assert unpacked.spreading[1] == pytest.approx(4.5)
    assert unpacked.trans_product[0] == pytest.approx(1.0)
    assert np.isnan(unpacked.trans_product[1])


def test_legacy_040_contract_is_preserved():
    """Additive outputs do not change the legacy result layout or ordering."""
    model = pd.DataFrame({
        "Depth": [0.0],
        "Vp": [2000.0],
        "Vs": [1000.0],
    })
    sources = np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]])
    receivers = np.array([[300.0, 400.0, 100.0], [600.0, 800.0, 100.0]])

    result = laytracer.trace_rays(sources, receivers, model, verbose=False)
    legacy_fields = [
        "travel_times",
        "rays",
        "ray_parameters",
        "tstar",
        "spreading",
        "trans_product",
        "source_phase",
    ]
    assert [field.name for field in fields(result)][:7] == legacy_fields
    assert result.travel_times.dtype == np.float64
    assert result.ray_parameters.dtype == np.float64
    assert result.travel_times.shape == (4,)
    assert result.ray_parameters.shape == (4,)
    assert isinstance(result.rays, list)
    assert all(ray.dtype == np.float64 and ray.shape == (2, 3) for ray in result.rays)
    assert result.tstar is None
    assert result.spreading is None
    assert result.trans_product is None
    assert result.complex_coefficient_product is None
    assert result.diagnostics is None
    assert result.sensitivities is None
    assert result.source_phase == "P"

    expected = np.array([500.0, 1000.0, np.sqrt(200.0**2 + 400.0**2),
                         np.sqrt(500.0**2 + 800.0**2)]) / 2000.0
    np.testing.assert_allclose(result.travel_times, expected, rtol=0.0, atol=1e-15)
    for flat_index, ray in enumerate(result.rays):
        source_index = flat_index // len(receivers)
        receiver_index = flat_index % len(receivers)
        np.testing.assert_array_equal(ray[0], sources[source_index])
        np.testing.assert_array_equal(ray[-1], receivers[receiver_index])

    scalar = laytracer.trace_rays(sources[0], receivers[0], model, source_phase="S")
    sequence = laytracer.trace_rays(
        sources[0], receivers[0], model, source_phase=["S"], verbose=False
    )
    assert isinstance(scalar, laytracer.TraceResult)
    assert list(sequence) == ["SV"]
    assert sequence["SV"].source_phase == "SV"

    parameter_names = list(inspect.signature(laytracer.trace_rays).parameters)
    assert parameter_names[:-1] == [
        "sources",
        "receivers",
        "velocity_df",
        "source_phase",
        "reflection",
        "refraction",
        "requested",
        "transcoef_method",
        "n_jobs",
        "backend",
        "sequential_limit",
        "rays_per_chunk",
        "tol",
        "max_iter",
        "verbose",
    ]
    assert parameter_names[-1] == "itinerary"


def test_explicit_itinerary_matches_legacy_reflection():
    """The ordered API is additive over the established reflection path."""
    model = _simple_model()
    source = np.array([0.0, 0.0, 100.0])
    receiver = np.array([1200.0, 300.0, 100.0])
    itinerary = laytracer.RayItinerary(
        "P", [laytracer.Interaction(1000.0, "reflect", "S")]
    )
    requested = {"travel_times", "rays", "ray_parameters"}

    explicit = laytracer.trace_rays(
        source, receiver, model, itinerary=itinerary,
        requested=requested, tol=1e-10, max_iter=50, verbose=False,
    )
    legacy = laytracer.trace_rays(
        source, receiver, model, reflection=[(1000.0, "S")],
        requested=requested, tol=1e-10, max_iter=50, verbose=False,
    )

    assert explicit.source_phase == "P"
    np.testing.assert_allclose(explicit.travel_times, legacy.travel_times, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(explicit.ray_parameters, legacy.ray_parameters, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(explicit.rays[0], legacy.rays[0], rtol=0.0, atol=1e-12)


def test_itinerary_rejects_legacy_interactions_and_unreachable_order():
    """An itinerary is exclusive and its vertical directions are explicit."""
    model = _simple_model()
    source = np.array([0.0, 0.0, 100.0])
    receiver = np.array([1000.0, 0.0, 100.0])
    itinerary = laytracer.RayItinerary(
        "P", [laytracer.Interaction(1000.0, "reflect", "P")]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        laytracer.trace_rays(
            source, receiver, model, itinerary=itinerary,
            reflection=[(1000.0, "P")], verbose=False,
        )

    unreachable = laytracer.RayItinerary(
        "P",
        [
            laytracer.Interaction(1000.0, "transmit", "P"),
            laytracer.Interaction(0.0, "transmit", "P"),
        ],
    )
    with pytest.raises(ValueError, match="not reachable"):
        laytracer.trace_rays(
            source, np.array([1000.0, 0.0, 2500.0]), model,
            itinerary=unreachable, verbose=False,
        )


def test_explicit_transmission_matches_legacy_mode_conversion():
    """Legacy ``refraction`` remains a wrapper for prescribed transmission."""
    model = _simple_model()
    source = np.array([0.0, 0.0, 100.0])
    receiver = np.array([1600.0, -200.0, 2500.0])
    itinerary = laytracer.RayItinerary(
        "P", [laytracer.Interaction(1000.0, "transmit", "SV")]
    )
    requested = {"travel_times", "rays", "ray_parameters"}

    explicit = laytracer.trace_rays(
        source,
        receiver,
        model,
        itinerary=itinerary,
        requested=requested,
        tol=1e-10,
        max_iter=50,
        verbose=False,
    )
    legacy = laytracer.trace_rays(
        source,
        receiver,
        model,
        refraction=[(1000.0, "SV")],
        requested=requested,
        tol=1e-10,
        max_iter=50,
        verbose=False,
    )

    np.testing.assert_allclose(explicit.travel_times, legacy.travel_times, atol=1e-12)
    np.testing.assert_allclose(explicit.ray_parameters, legacy.ray_parameters, atol=1e-15)
    np.testing.assert_allclose(explicit.rays[0], legacy.rays[0], atol=1e-12)


def test_converted_reflection_obeys_reverse_itinerary_reciprocity():
    """Reversing endpoint phases and conversion direction preserves kinematics."""
    model = _simple_model()
    source = np.array([100.0, -300.0, 200.0])
    receiver = np.array([2400.0, 500.0, 600.0])
    forward = laytracer.RayItinerary(
        "P", [laytracer.Interaction(2000.0, "reflect", "SV")]
    )
    reverse = laytracer.RayItinerary(
        "SV", [laytracer.Interaction(2000.0, "reflect", "P")]
    )
    requested = {"travel_times", "ray_parameters", "rays"}
    result_forward = laytracer.trace_rays(
        source, receiver, model, itinerary=forward, requested=requested,
        tol=1e-10, max_iter=50, verbose=False,
    )
    result_reverse = laytracer.trace_rays(
        receiver, source, model, itinerary=reverse, requested=requested,
        tol=1e-10, max_iter=50, verbose=False,
    )

    np.testing.assert_allclose(
        result_forward.travel_times, result_reverse.travel_times, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result_forward.ray_parameters, result_reverse.ray_parameters, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        result_forward.rays[0], result_reverse.rays[0][::-1], rtol=0.0, atol=1e-10
    )


def test_explicit_converted_multiple_keeps_endpoints_and_phase_topology():
    """A prescribed converted down-and-up path supports repeated traversals."""
    model = _simple_model()
    source = np.array([0.0, 0.0, 100.0])
    receiver = np.array([1800.0, 600.0, 100.0])
    itinerary = laytracer.RayItinerary(
        "P",
        [
            laytracer.Interaction(1000.0, "transmit", "SV"),
            laytracer.Interaction(2000.0, "reflect", "SV"),
            laytracer.Interaction(1000.0, "transmit", "P"),
        ],
    )
    result = laytracer.trace_rays(
        source,
        receiver,
        model,
        itinerary=itinerary,
        requested={"travel_times", "rays", "ray_parameters", "sensitivities"},
        tol=1e-10,
        max_iter=50,
        verbose=False,
    )

    assert np.isfinite(result.travel_times[0])
    assert np.isfinite(result.ray_parameters[0])
    np.testing.assert_allclose(result.rays[0][0], source, atol=1e-12)
    np.testing.assert_allclose(result.rays[0][-1], receiver, atol=1e-9)
    assert result.sensitivities[0].valid
    np.testing.assert_array_equal(result.sensitivities[0].vp_layer_indices, [0])
    np.testing.assert_array_equal(result.sensitivities[0].vs_layer_indices, [1])


def _tile_geometry():
    """Three sources and seven receivers, so tiles of four rays are ragged."""
    sources = np.array([
        [0.0, 0.0, 500.0],
        [300.0, -200.0, 1500.0],
        [-400.0, 100.0, 2500.0],
    ])
    receivers = np.array([
        [3000.0, 0.0, 0.0],
        [5000.0, 1000.0, 100.0],
        [-2000.0, 700.0, 800.0],
        [1000.0, -3000.0, 1200.0],
        [0.0, 0.0, 1800.0],
        [2500.0, 2500.0, 2200.0],
        [-1500.0, -500.0, 2900.0],
    ])
    return sources, receivers


def _assert_same_trace(actual, expected):
    for name in (
        "travel_times",
        "ray_parameters",
        "tstar",
        "spreading",
        "trans_product",
        "complex_coefficient_product",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.source_phase == expected.source_phase
    assert len(actual.rays) == len(expected.rays)
    for ray, expected_ray in zip(actual.rays, expected.rays):
        np.testing.assert_array_equal(ray, expected_ray)
    assert [repr(d) for d in actual.diagnostics] == [repr(d) for d in expected.diagnostics]
    assert len(actual.sensitivities) == len(expected.sensitivities)
    for sensitivity, expected_sensitivity in zip(actual.sensitivities, expected.sensitivities):
        for name, value in vars(expected_sensitivity).items():
            np.testing.assert_array_equal(getattr(sensitivity, name), value)


@pytest.mark.parametrize("source_phase", ["P", ["P", "SV", "SH"]])
@pytest.mark.parametrize(
    "n_jobs, backend, rays_per_chunk",
    [(2, "threading", 4), (2, "threading", None), (1, "loky", 4), (2, "loky", 4)],
)
def test_tiled_tracing_matches_sequential(source_phase, n_jobs, backend, rays_per_chunk):
    """Tiles copied into the output reproduce the sequential result exactly."""
    df = _simple_model()
    sources, receivers = _tile_geometry()
    kwargs = dict(source_phase=source_phase, requested=TRACE_OUTPUTS, verbose=False)
    expected = laytracer.trace_rays(sources, receivers, df, **kwargs)
    try:
        actual = laytracer.trace_rays(
            sources,
            receivers,
            df,
            n_jobs=n_jobs,
            backend=backend,
            sequential_limit=0,
            rays_per_chunk=rays_per_chunk,
            **kwargs,
        )
    except PermissionError:
        pytest.skip("Parallel worker creation is blocked in this environment")

    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        for phase in expected:
            _assert_same_trace(actual[phase], expected[phase])
    else:
        _assert_same_trace(actual, expected)


def test_loky_workers_trace_with_one_blas_thread():
    """Loky workers get one BLAS thread, since tracing makes no BLAS calls."""
    df = _simple_model()
    sources, receivers = _tile_geometry()
    try:
        laytracer.trace_rays(
            sources, receivers, df, n_jobs=2, backend="loky",
            sequential_limit=0, rays_per_chunk=4, verbose=False,
        )
    except PermissionError:
        pytest.skip("Parallel worker creation is blocked in this environment")

    # The reusable loky workers stay alive after the call; read their environment.
    blas_threads = []
    for child in psutil.Process().children(recursive=True):
        try:
            blas_threads.append(child.environ().get("OPENBLAS_NUM_THREADS"))
        except psutil.Error:
            continue
    assert blas_threads.count("1") >= 2


def test_tile_worker_returns_compact_columns():
    """A task returns NumPy columns, not one Python object per ray."""
    df = _simple_model()
    sources = np.column_stack([np.zeros(4), np.zeros(4), np.linspace(500.0, 2500.0, 4)])
    receivers = np.column_stack(
        [np.linspace(-3000.0, 3000.0, 100), np.zeros(100), np.full(100, 100.0)]
    )
    kw = dict(
        ma=ModelArrays.from_dataframe(df),
        source_phases=("P", "SV"),
        return_dict=True,
        refl_list=[],
        refr_list=[],
        ordered_interactions=None,
        need_rays=False,
        need_ray_parameters=False,
        need_tstar=False,
        need_spreading=False,
        need_trans_product=False,
        need_complex_coefficient_product=False,
        need_diagnostics=False,
        need_sensitivities=False,
        transcoef_method="standard",
        tol=1e-4,
        max_iter=10,
    )
    tile = (0, 4, 0, 100)

    returned_tile, columns = _trace_batch(tile, sources, receivers, kw)

    assert returned_tile == tile
    n_rays = len(sources) * len(receivers)
    for phase in ("P", "SV"):
        assert columns[phase]["tt"].dtype == np.float64
        assert columns[phase]["tt"].shape == (n_rays,)
        assert np.all(np.isfinite(columns[phase]["tt"]))
    # About 8 bytes per ray and phase plus a fixed pickle overhead; the 0.4
    # per-ray tuples pickled to roughly 60 bytes per ray for two phases.
    assert len(pickle.dumps(columns)) < 2 * 8 * n_rays + 2048


def test_warns_when_output_exceeds_available_memory(monkeypatch):
    """The estimated output size is checked against available memory up front."""
    monkeypatch.setattr(
        laytracer.api.psutil, "virtual_memory", lambda: SimpleNamespace(available=1)
    )
    df = _simple_model()
    sources, receivers = _tile_geometry()
    with pytest.warns(RuntimeWarning, match="memory is available"):
        laytracer.trace_rays(
            sources, receivers, df, n_jobs=1, sequential_limit=0, verbose=False
        )


def test_verbose_reports_task_plan_and_progress(capsys):
    """Verbose tracing prints the task plan and reaches 100 percent."""
    df = _simple_model()
    sources, receivers = _tile_geometry()
    laytracer.trace_rays(
        sources, receivers, df, n_jobs=1, sequential_limit=0, rays_per_chunk=4
    )
    out = capsys.readouterr().out
    assert "Tracing 21 rays in 8 tasks of up to 4 rays" in out
    assert "100% done" in out

