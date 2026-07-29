"""Analytic fixed-topology sensitivity tests."""

import numpy as np
import pandas as pd
import pytest

import laytracer


def _trace(source, receiver, model, **kwargs):
    return laytracer.trace_rays(
        source,
        receiver,
        model,
        tol=1e-11,
        max_iter=50,
        verbose=False,
        **kwargs,
    )


def test_homogeneous_velocity_and_endpoint_sensitivities():
    """The sparse operator reduces to straight-ray derivatives."""
    velocity = 2000.0
    model = pd.DataFrame({"Depth": [0.0], "Vp": [velocity], "Vs": [1000.0]})
    source = np.array([0.0, 50.0, 100.0])
    receiver = np.array([1000.0, 250.0, 500.0])
    delta = receiver - source
    distance = np.linalg.norm(delta)

    result = _trace(
        source,
        receiver,
        model,
        requested={"travel_times", "sensitivities", "diagnostics"},
    )
    sensitivity = result.sensitivities[0]

    assert result.ray_parameters is None
    assert result.diagnostics[0].converged
    assert sensitivity.valid
    np.testing.assert_array_equal(sensitivity.vp_layer_indices, [0])
    assert sensitivity.vs_layer_indices.size == 0
    assert sensitivity.dtravel_time_dvp[0] == pytest.approx(
        -distance / velocity**2, rel=1e-12
    )
    ray_parameter = np.hypot(delta[0], delta[1]) / (velocity * distance)
    assert sensitivity.dray_parameter_dvp[0] == pytest.approx(
        -ray_parameter / velocity, rel=1e-12
    )
    np.testing.assert_allclose(
        sensitivity.dtravel_time_dsource,
        -delta / (velocity * distance),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        sensitivity.dtravel_time_dreceiver,
        delta / (velocity * distance),
        rtol=1e-12,
        atol=1e-15,
    )


def test_mixed_reflection_sensitivities_match_finite_differences():
    """Repeated mixed-phase legs aggregate by physical model parameter."""
    model = pd.DataFrame({
        "Depth": [0.0, 1000.0, 2000.0],
        "Vp": [2000.0, 3000.0, 4000.0],
        "Vs": [1000.0, 1600.0, 2200.0],
    })
    source = np.array([0.0, 0.0, 100.0])
    receiver = np.array([1200.0, 300.0, 100.0])
    itinerary = laytracer.RayItinerary(
        "P", [laytracer.Interaction(1000.0, "reflect", "SV")]
    )
    result = _trace(
        source,
        receiver,
        model,
        itinerary=itinerary,
        requested={"travel_times", "ray_parameters", "sensitivities"},
    )
    sensitivity = result.sensitivities[0]

    def finite_difference(column, step):
        plus = model.copy()
        minus = model.copy()
        plus.loc[0, column] += step
        minus.loc[0, column] -= step
        requested = {"travel_times", "ray_parameters"}
        result_plus = _trace(
            source, receiver, plus, itinerary=itinerary, requested=requested
        )
        result_minus = _trace(
            source, receiver, minus, itinerary=itinerary, requested=requested
        )
        return (
            (result_plus.travel_times[0] - result_minus.travel_times[0]) / (2 * step),
            (result_plus.ray_parameters[0] - result_minus.ray_parameters[0]) / (2 * step),
        )

    dtime_vp, dparameter_vp = finite_difference("Vp", 1e-2)
    dtime_vs, dparameter_vs = finite_difference("Vs", 1e-2)
    assert sensitivity.dtravel_time_dvp[0] == pytest.approx(dtime_vp, rel=1e-9)
    assert sensitivity.dray_parameter_dvp[0] == pytest.approx(dparameter_vp, rel=1e-9)
    assert sensitivity.dtravel_time_dvs[0] == pytest.approx(dtime_vs, rel=1e-9)
    assert sensitivity.dray_parameter_dvs[0] == pytest.approx(dparameter_vs, rel=1e-9)

    depth_step = 1e-2
    plus = model.copy()
    minus = model.copy()
    plus.loc[1, "Depth"] += depth_step
    minus.loc[1, "Depth"] -= depth_step
    plus_itinerary = laytracer.RayItinerary(
        "P", [laytracer.Interaction(1000.0 + depth_step, "reflect", "SV")]
    )
    minus_itinerary = laytracer.RayItinerary(
        "P", [laytracer.Interaction(1000.0 - depth_step, "reflect", "SV")]
    )
    requested = {"travel_times", "ray_parameters"}
    result_plus = _trace(
        source, receiver, plus, itinerary=plus_itinerary, requested=requested
    )
    result_minus = _trace(
        source, receiver, minus, itinerary=minus_itinerary, requested=requested
    )
    dtime_depth = (
        result_plus.travel_times[0] - result_minus.travel_times[0]
    ) / (2 * depth_step)
    dparameter_depth = (
        result_plus.ray_parameters[0] - result_minus.ray_parameters[0]
    ) / (2 * depth_step)

    np.testing.assert_array_equal(sensitivity.interface_indices, [1])
    assert sensitivity.dtravel_time_dinterface_depths[0] == pytest.approx(
        dtime_depth, rel=1e-9
    )
    assert sensitivity.dray_parameter_dinterface_depths[0] == pytest.approx(
        dparameter_depth, rel=1e-9
    )


def test_chunked_sensitivities_keep_source_major_order():
    """Optional object outputs follow the legacy flattened ray order."""
    model = pd.DataFrame({"Depth": [0.0], "Vp": [2000.0], "Vs": [1000.0]})
    sources = np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]])
    receivers = np.array([[300.0, 400.0, 100.0], [600.0, 800.0, 100.0]])
    result = _trace(
        sources,
        receivers,
        model,
        requested={"travel_times", "sensitivities", "diagnostics"},
        n_jobs=2,
        backend="threading",
        sequential_limit=0,
        rays_per_chunk=1,
    )

    assert len(result.sensitivities) == 4
    assert len(result.diagnostics) == 4
    for flat_index, sensitivity in enumerate(result.sensitivities):
        source_index = flat_index // len(receivers)
        receiver_index = flat_index % len(receivers)
        distance = np.linalg.norm(
            receivers[receiver_index] - sources[source_index]
        )
        assert sensitivity.dtravel_time_dvp[0] == pytest.approx(
            -distance / 2000.0**2, rel=1e-12
        )
        assert result.diagnostics[flat_index].converged
