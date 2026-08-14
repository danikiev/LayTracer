"""Reusable traveltime approximation tests."""

import numpy as np
import pandas as pd
import pytest

import laytracer


def _model():
    return pd.DataFrame({
        "Depth": [0.0, 700.0, 1400.0],
        "Vp": [2600.0, 3400.0, 4300.0],
        "Vs": [1450.0, 1950.0, 2500.0],
    })


def test_linearized_ray_change_matches_manual_sparse_products():
    """The public helper applies all sparse derivative families correctly."""
    model = _model()
    source = np.array([0.0, 0.0, 100.0])
    receiver = np.array([1800.0, 0.0, 100.0])
    itinerary = laytracer.RayItinerary(
        "P", [laytracer.Interaction(1400.0, "reflect", "SV")]
    )
    traced = laytracer.trace_rays(
        source,
        receiver,
        model,
        itinerary=itinerary,
        requested={"travel_times", "sensitivities"},
        n_jobs=1,
        verbose=False,
    )
    sensitivity = traced.sensitivities[0]
    delta_vp = np.array([10.0, -20.0, 0.0])
    delta_vs = np.array([-5.0, 15.0, 0.0])
    delta_depth = np.array([0.0, 3.0, -4.0])
    delta_source = np.array([2.0, 0.0, -1.0])
    delta_receiver = np.array([-3.0, 0.0, 2.0])

    predicted = laytracer.linearized_ray_change(
        sensitivity,
        delta_vp=delta_vp,
        delta_vs=delta_vs,
        delta_interface_depths=delta_depth,
        delta_source=delta_source,
        delta_receiver=delta_receiver,
    )
    expected_time = (
        np.dot(sensitivity.dtravel_time_dvp, delta_vp[sensitivity.vp_layer_indices])
        + np.dot(sensitivity.dtravel_time_dvs, delta_vs[sensitivity.vs_layer_indices])
        + np.dot(
            sensitivity.dtravel_time_dinterface_depths,
            delta_depth[sensitivity.interface_indices],
        )
        + np.dot(sensitivity.dtravel_time_dsource, delta_source)
        + np.dot(sensitivity.dtravel_time_dreceiver, delta_receiver)
    )
    expected_parameter = (
        np.dot(sensitivity.dray_parameter_dvp, delta_vp[sensitivity.vp_layer_indices])
        + np.dot(sensitivity.dray_parameter_dvs, delta_vs[sensitivity.vs_layer_indices])
        + np.dot(
            sensitivity.dray_parameter_dinterface_depths,
            delta_depth[sensitivity.interface_indices],
        )
        + np.dot(sensitivity.dray_parameter_dsource, delta_source)
        + np.dot(sensitivity.dray_parameter_dreceiver, delta_receiver)
    )
    assert predicted.delta_travel_time == pytest.approx(expected_time)
    assert predicted.delta_ray_parameter == pytest.approx(expected_parameter)

    with pytest.raises(ValueError, match="delta_vp"):
        laytracer.linearized_ray_change(sensitivity, delta_vp=np.ones(1))
    with pytest.raises(ValueError, match="finite"):
        laytracer.linearized_ray_change(
            sensitivity,
            delta_vp=np.array([np.nan, 0.0, 0.0]),
        )


def test_select_anchors_is_deterministic_grouped_and_covering():
    """Irregular point clouds are covered independently within each group."""
    points = np.array([
        [0.0, 0.0, 100.0],
        [40.0, 10.0, 100.0],
        [105.0, -5.0, 100.0],
        [105.0, -5.0, 100.0],
        [0.0, 0.0, 800.0],
        [60.0, 25.0, 820.0],
        [160.0, 30.0, 840.0],
    ])
    groups = np.array([0, 0, 0, 0, 1, 1, 1])
    first = laytracer.select_anchors(points, 80.0, groups=groups)
    second = laytracer.select_anchors(points, 80.0, groups=groups)

    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.assignments, second.assignments)
    assert np.all(first.distances <= 80.0 + 1e-12)
    assigned_indices = first.indices[first.assignments]
    np.testing.assert_array_equal(groups[assigned_indices], groups)
    assert first.assignments[2] == first.assignments[3]


def test_direct_approximator_preserves_order_and_is_exact_at_anchors():
    """Fitted direct-ray predictions use fewer exact rays and source-major order."""
    model = pd.DataFrame({"Depth": [0.0], "Vp": [2000.0], "Vs": [1000.0]})
    sources = np.column_stack([
        np.arange(0.0, 500.0, 50.0),
        np.zeros(10),
        np.full(10, 100.0),
    ])
    receivers = np.array([[1000.0, 0.0, 100.0], [1200.0, 0.0, 100.0]])
    approximator = laytracer.TravelTimeApproximator.fit(
        sources,
        receivers,
        model,
        source_max_distance=100.0,
        receiver_max_distance=None,
        n_jobs=1,
    )
    prediction = approximator.predict()
    exact = laytracer.trace_rays(
        sources,
        receivers,
        model,
        requested={"travel_times"},
        n_jobs=1,
        verbose=False,
    )

    assert prediction.valid_mask.all()
    assert approximator.exact_ray_count < len(exact.travel_times)
    np.testing.assert_allclose(prediction.travel_times, exact.travel_times, atol=1e-14)
    np.testing.assert_allclose(
        prediction.travel_time_matrix,
        exact.travel_times.reshape(len(sources), len(receivers)),
    )
    assert prediction.anchor_ray_index_matrix.shape == (len(sources), len(receivers))
    assert prediction.source_anchor_distance_matrix.shape == (len(sources), len(receivers))
    assert prediction.receiver_anchor_distance_matrix.shape == (len(sources), len(receivers))
    anchor_prediction = approximator.predict(
        approximator.anchor_sources,
        approximator.anchor_receivers,
    )
    np.testing.assert_allclose(
        anchor_prediction.travel_times,
        approximator.anchor_trace.travel_times,
        atol=1e-14,
    )


def test_direct_topology_refinement_handles_both_sides_of_receiver():
    """Targets above and below a receiver never share the wrong direct branch."""
    model = _model()
    sources = np.array([
        [100.0, 0.0, 125.0],
        [100.0, 0.0, 325.0],
        [100.0, 0.0, 525.0],
        [100.0, 0.0, 675.0],
    ])
    receivers = np.array([[0.0, 0.0, 400.0]])
    approximator = laytracer.TravelTimeApproximator.fit(
        sources,
        receivers,
        model,
        source_max_distance=300.0,
        receiver_max_distance=None,
        n_jobs=1,
    )
    prediction = approximator.predict()
    exact = laytracer.trace_rays(
        sources, receivers, model, requested={"travel_times"}, n_jobs=1, verbose=False
    )

    assert prediction.valid_mask.all()
    assert np.sqrt(np.mean((prediction.travel_times - exact.travel_times) ** 2)) < 5e-3
    chosen_source_positions = (
        prediction.anchor_ray_indices // len(approximator.anchor_receivers)
    )
    chosen_source_depths = approximator.anchor_sources[chosen_source_positions, 2]
    target_depths = np.repeat(sources[:, 2], len(receivers))
    assert np.all((chosen_source_depths - 400.0) * (target_depths - 400.0) > 0.0)


def test_itinerary_prediction_masks_unreachable_and_outside_targets():
    """Unsupported itinerary and extrapolation targets are reported, not traced."""
    model = _model()
    sources = np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]])
    receivers = np.array([[1200.0, 0.0, 100.0], [1400.0, 0.0, 100.0]])
    itinerary = laytracer.RayItinerary(
        "P", [laytracer.Interaction(1400.0, "reflect", "SV")]
    )
    approximator = laytracer.TravelTimeApproximator.fit(
        sources,
        receivers,
        model,
        source_max_distance=100.0,
        receiver_max_distance=250.0,
        itinerary=itinerary,
        n_jobs=1,
    )

    unreachable = approximator.predict(
        sources[0],
        np.array([1200.0, 0.0, 1600.0]),
    )
    assert not unreachable.valid_mask[0]
    assert unreachable.reasons[0] == "unreachable_topology"
    outside = approximator.predict(
        np.array([5000.0, 0.0, 100.0]),
        receivers[0],
    )
    assert not outside.valid_mask[0]
    assert outside.reasons[0] == "outside_anchor_distance"


def test_prediction_reports_missing_topology_and_invalid_sensitivity():
    """Reachable unsupported branches and unusable anchors have distinct reasons."""
    model = pd.DataFrame({"Depth": [0.0], "Vp": [2000.0], "Vs": [1000.0]})
    approximator = laytracer.TravelTimeApproximator.fit(
        np.array([[0.0, 0.0, 100.0]]),
        np.array([[1000.0, 0.0, 300.0]]),
        model,
        source_max_distance=500.0,
        receiver_max_distance=None,
        n_jobs=1,
    )

    missing = approximator.predict(
        np.array([[0.0, 0.0, 500.0]]),
        np.array([[1000.0, 0.0, 300.0]]),
    )
    assert missing.reasons[0] == "no_topology_match"

    approximator.anchor_trace.sensitivities[0].valid = False
    invalid = approximator.predict()
    assert invalid.reasons[0] == "invalid_anchor_sensitivity"
    assert np.isnan(invalid.travel_times[0])


def test_three_layer_source_grid_reduces_exact_ray_count_with_bounded_error():
    """A small vertical-well table reproduces the Example 06 use case."""
    model = _model()
    source_x = np.arange(0.0, 1000.0 + 100.0, 100.0)
    source_z = np.arange(50.0, 1200.0, 100.0)
    grid_x, grid_z = np.meshgrid(source_x, source_z)
    sources = np.column_stack([
        grid_x.ravel(),
        np.zeros(grid_x.size),
        grid_z.ravel(),
    ])
    receivers = np.column_stack([
        np.zeros(5),
        np.zeros(5),
        np.arange(100.0, 600.0, 100.0),
    ])
    approximator = laytracer.TravelTimeApproximator.fit(
        sources,
        receivers,
        model,
        source_max_distance=180.0,
        receiver_max_distance=None,
        n_jobs=1,
    )
    prediction = approximator.predict()
    exact = laytracer.trace_rays(
        sources, receivers, model, requested={"travel_times"}, n_jobs=1, verbose=False
    )
    error_ms = 1e3 * (prediction.travel_times - exact.travel_times)

    assert prediction.valid_mask.all()
    assert approximator.exact_ray_count < len(exact.travel_times)
    assert np.sqrt(np.mean(error_ms**2)) < 5.0
