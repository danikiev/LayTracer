r"""Topology-safe local approximation of dense traveltime tables.

This module turns analytic fixed-topology endpoint derivatives into reusable
traveltime predictors.  It does not estimate approximation error or retrace
unsupported targets automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .api import RayItinerary, TraceResult, trace_rays
from .model import ModelArrays, normalize_phase
from .sensitivity import RaySensitivity, compile_path_legs


_REASON_OK = "ok"
_REASON_UNREACHABLE = "unreachable_topology"
_REASON_NO_MATCH = "no_topology_match"
_REASON_OUTSIDE = "outside_anchor_distance"
_REASON_INVALID_SENSITIVITY = "invalid_anchor_sensitivity"


@dataclass(frozen=True)
class LinearizedRayChange:
    """First-order changes predicted from one ray sensitivity record.

    Attributes
    ----------
    delta_travel_time : float
        Predicted traveltime change (s).
    delta_ray_parameter : float
        Predicted physical horizontal-slowness change (s/m).
    """

    delta_travel_time: float
    delta_ray_parameter: float


@dataclass(frozen=True)
class AnchorSelection:
    """Deterministic point-cloud anchor selection and assignments.

    Attributes
    ----------
    indices : numpy.ndarray
        Indices of selected anchors in the input point array.
    points : numpy.ndarray
        Selected anchor coordinates, shape ``(n_anchors, 3)``.
    assignments : numpy.ndarray
        Anchor-position assigned to each input point.
    distances : numpy.ndarray
        Euclidean distance from each point to its assigned anchor (m).
    """

    indices: np.ndarray
    points: np.ndarray
    assignments: np.ndarray
    distances: np.ndarray


@dataclass(frozen=True)
class TravelTimePrediction:
    """Predicted traveltimes and anchor-assignment diagnostics.

    All one-dimensional arrays use source-major ordering: every receiver for
    source zero, followed by every receiver for source one, and so on.

    Attributes
    ----------
    travel_times : numpy.ndarray
        Predicted traveltimes (s); invalid entries are ``NaN``.
    valid_mask : numpy.ndarray
        Whether each prediction is valid.
    reasons : numpy.ndarray
        Reason code for each prediction.
    anchor_ray_indices : numpy.ndarray
        Flat index of the selected exact anchor ray, or ``-1`` if invalid.
    source_anchor_distances, receiver_anchor_distances : numpy.ndarray
        Endpoint distances to the selected anchor ray (m), or ``NaN`` if
        invalid.
    n_sources, n_receivers : int
        Dimensions used by the matrix-view properties.
    """

    travel_times: np.ndarray
    valid_mask: np.ndarray
    reasons: np.ndarray
    anchor_ray_indices: np.ndarray
    source_anchor_distances: np.ndarray
    receiver_anchor_distances: np.ndarray
    n_sources: int
    n_receivers: int

    @property
    def travel_time_matrix(self) -> np.ndarray:
        """Return traveltimes with shape ``(n_sources, n_receivers)``."""
        return self.travel_times.reshape(self.n_sources, self.n_receivers)

    @property
    def valid_matrix(self) -> np.ndarray:
        """Return the validity mask with shape ``(n_sources, n_receivers)``."""
        return self.valid_mask.reshape(self.n_sources, self.n_receivers)

    @property
    def reason_matrix(self) -> np.ndarray:
        """Return reason codes with shape ``(n_sources, n_receivers)``."""
        return self.reasons.reshape(self.n_sources, self.n_receivers)

    @property
    def anchor_ray_index_matrix(self) -> np.ndarray:
        """Return selected anchor-ray indices in source-major matrix form."""
        return self.anchor_ray_indices.reshape(self.n_sources, self.n_receivers)

    @property
    def source_anchor_distance_matrix(self) -> np.ndarray:
        """Return source-to-anchor distances in source-major matrix form."""
        return self.source_anchor_distances.reshape(self.n_sources, self.n_receivers)

    @property
    def receiver_anchor_distance_matrix(self) -> np.ndarray:
        """Return receiver-to-anchor distances in source-major matrix form."""
        return self.receiver_anchor_distances.reshape(self.n_sources, self.n_receivers)


def _normalize_points(points, name: str) -> np.ndarray:
    """Return a finite ``(N, 3)`` coordinate array."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        array = np.atleast_2d(array)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, 3) or (3,).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates.")
    return array


def _delta_for_indices(values, indices: np.ndarray, name: str) -> np.ndarray:
    """Extract a sparse derivative-compatible perturbation array."""
    if values is None:
        return np.zeros(indices.size, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional full-model array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite perturbations.")
    if indices.size and array.size <= int(np.max(indices)):
        raise ValueError(f"{name} does not contain every referenced model index.")
    return array[indices]


def _endpoint_delta(values, name: str) -> np.ndarray:
    """Return one three-component endpoint perturbation."""
    if values is None:
        return np.zeros(3, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-component vector.")
    return array


def linearized_ray_change(
    sensitivity: RaySensitivity,
    *,
    delta_vp=None,
    delta_vs=None,
    delta_interface_depths=None,
    delta_source=None,
    delta_receiver=None,
) -> LinearizedRayChange:
    r"""Predict first-order output changes for one fixed-topology ray.

    Parameters
    ----------
    sensitivity : RaySensitivity
        Valid sparse sensitivity record returned by :func:`trace_rays`.
    delta_vp, delta_vs : array-like, optional
        Full model-row-indexed velocity perturbations (m/s).
    delta_interface_depths : array-like, optional
        Full model-row-indexed interface-depth perturbations (m).
    delta_source, delta_receiver : array-like, optional
        Three-component endpoint-coordinate perturbations (m).

    Returns
    -------
    LinearizedRayChange
        First-order traveltime and physical ray-parameter changes.

    Notes
    -----
    Omitted perturbations are treated as zero. The calculation is valid only
    while the ray topology represented by ``sensitivity`` remains unchanged.
    """
    if not isinstance(sensitivity, RaySensitivity):
        raise TypeError("sensitivity must be a RaySensitivity object.")
    if not sensitivity.valid:
        reason = sensitivity.reason or "unspecified reason"
        raise ValueError(f"Cannot use an invalid sensitivity: {reason}")

    vp = _delta_for_indices(delta_vp, sensitivity.vp_layer_indices, "delta_vp")
    vs = _delta_for_indices(delta_vs, sensitivity.vs_layer_indices, "delta_vs")
    interfaces = _delta_for_indices(
        delta_interface_depths,
        sensitivity.interface_indices,
        "delta_interface_depths",
    )
    source = _endpoint_delta(delta_source, "delta_source")
    receiver = _endpoint_delta(delta_receiver, "delta_receiver")

    delta_time = (
        np.dot(sensitivity.dtravel_time_dvp, vp)
        + np.dot(sensitivity.dtravel_time_dvs, vs)
        + np.dot(sensitivity.dtravel_time_dinterface_depths, interfaces)
        + np.dot(sensitivity.dtravel_time_dsource, source)
        + np.dot(sensitivity.dtravel_time_dreceiver, receiver)
    )
    delta_parameter = (
        np.dot(sensitivity.dray_parameter_dvp, vp)
        + np.dot(sensitivity.dray_parameter_dvs, vs)
        + np.dot(sensitivity.dray_parameter_dinterface_depths, interfaces)
        + np.dot(sensitivity.dray_parameter_dsource, source)
        + np.dot(sensitivity.dray_parameter_dreceiver, receiver)
    )
    return LinearizedRayChange(float(delta_time), float(delta_parameter))


def _stable_groups(groups, n_points: int) -> list[np.ndarray]:
    """Return group indices in first-occurrence order."""
    if groups is None:
        return [np.arange(n_points, dtype=np.int64)]
    labels = np.asarray(groups)
    if labels.ndim != 1 or labels.size != n_points:
        raise ValueError("groups must be one-dimensional with one label per point.")
    grouped: dict[object, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        try:
            grouped.setdefault(label, []).append(index)
        except TypeError as exc:
            raise ValueError("groups must contain hashable labels.") from exc
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def select_anchors(points, max_distance: float, groups=None) -> AnchorSelection:
    """Select deterministic farthest-point anchors covering a point cloud.

    Parameters
    ----------
    points : array-like
        Finite point coordinates, shape ``(n_points, 3)``.
    max_distance : float
        Maximum Euclidean point-to-anchor distance (m).
    groups : array-like, optional
        One group label per point. Coverage is constructed independently in
        each group, which can prevent anchors from crossing model layers.

    Returns
    -------
    AnchorSelection
        Selected indices and coordinates plus nearest-anchor assignments.

    Notes
    -----
    Selection begins at the lexicographically smallest point in each group,
    then repeatedly adds the farthest uncovered point. Ties are resolved by
    original point index, making the result deterministic.
    """
    coordinates = _normalize_points(points, "points")
    max_distance = float(max_distance)
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("max_distance must be positive and finite.")

    selected: list[int] = []
    tolerance = 1e-12 * max(1.0, max_distance)
    for group_indices in _stable_groups(groups, len(coordinates)):
        group_points = coordinates[group_indices]
        order = np.lexsort((group_indices, group_points[:, 2], group_points[:, 1], group_points[:, 0]))
        first = int(group_indices[order[0]])
        group_selected = [first]
        nearest = np.linalg.norm(group_points - coordinates[first], axis=1)
        while float(np.max(nearest)) > max_distance + tolerance:
            farthest_distance = float(np.max(nearest))
            candidates = np.flatnonzero(np.isclose(nearest, farthest_distance, rtol=0.0, atol=tolerance))
            next_index = int(group_indices[candidates[np.argmin(group_indices[candidates])]])
            group_selected.append(next_index)
            distance = np.linalg.norm(group_points - coordinates[next_index], axis=1)
            nearest = np.minimum(nearest, distance)
        selected.extend(group_selected)

    indices = np.asarray(selected, dtype=np.int64)
    anchor_points = coordinates[indices]
    assignments = np.empty(len(coordinates), dtype=np.int64)
    distances = np.empty(len(coordinates), dtype=np.float64)
    group_arrays = _stable_groups(groups, len(coordinates))
    for group_indices in group_arrays:
        if groups is None:
            group_anchor_positions = np.arange(len(indices), dtype=np.int64)
        else:
            group_labels = np.asarray(groups)
            label = group_labels[group_indices[0]]
            group_anchor_positions = np.flatnonzero(group_labels[indices] == label)
        pair_distances = np.linalg.norm(
            coordinates[group_indices, None, :] - anchor_points[group_anchor_positions][None, :, :],
            axis=2,
        )
        nearest_positions = np.argmin(pair_distances, axis=1)
        assignments[group_indices] = group_anchor_positions[nearest_positions]
        distances[group_indices] = pair_distances[np.arange(len(group_indices)), nearest_positions]
    return AnchorSelection(indices, anchor_points.copy(), assignments, distances)


def _layer_indices(depths: np.ndarray, model_depths: np.ndarray) -> np.ndarray:
    """Return physical layer indices for endpoint depths."""
    return np.maximum(np.searchsorted(model_depths, depths, side="right") - 1, 0)


def _topology_key(
    model: ModelArrays,
    source: np.ndarray,
    receiver: np.ndarray,
    source_phase: str,
    itinerary: RayItinerary | None,
):
    """Return an ordered path-leg key, or ``None`` if unreachable."""
    source_depth = float(source[2])
    receiver_depth = float(receiver[2])
    current_depth = source_depth
    current_phase = source_phase
    direction = None
    segments: list[dict] = []
    tolerance = 1e-9

    interactions = itinerary.interactions if itinerary is not None else ()
    for interaction in interactions:
        delta = interaction.depth - current_depth
        if abs(delta) <= tolerance:
            return None
        incoming_direction = 1 if delta > 0.0 else -1
        if direction is not None and incoming_direction != direction:
            return None
        direction = incoming_direction
        segments.append({
            "start_z": current_depth,
            "end_z": interaction.depth,
            "phase": current_phase,
        })
        current_depth = interaction.depth
        current_phase = interaction.outgoing_phase
        if interaction.kind == "reflect":
            direction *= -1

    receiver_delta = receiver_depth - current_depth
    if abs(receiver_delta) > tolerance:
        receiver_direction = 1 if receiver_delta > 0.0 else -1
        if direction is not None and receiver_direction != direction:
            return None
        segments.append({
            "start_z": current_depth,
            "end_z": receiver_depth,
            "phase": current_phase,
        })

    legs = compile_path_legs(model, segments)
    if legs:
        return tuple((leg.layer_index, leg.phase, int(leg.direction)) for leg in legs)

    horizontal_distance = float(np.linalg.norm(receiver[:2] - source[:2]))
    layer = int(_layer_indices(np.array([source_depth]), model.depths)[0])
    if horizontal_distance > tolerance:
        return ((layer, current_phase, 0),)
    return ((layer, current_phase, "coincident"),)


def _distance_allowed(distance: np.ndarray, limit: float | None) -> np.ndarray:
    """Return whether endpoint distances satisfy one anchor limit."""
    if limit is None:
        return distance <= 1e-9
    return distance <= limit + 1e-12 * max(1.0, limit)


def _normalized_distance(distance: np.ndarray, limit: float | None) -> np.ndarray:
    """Normalize endpoint distances for joint anchor ranking."""
    if limit is None:
        return np.zeros_like(distance)
    return distance / limit


def _extend_anchor_cover(
    points: np.ndarray,
    target_indices: np.ndarray,
    selected: set[int],
    max_distance: float,
) -> None:
    """Extend an anchor set to cover one topology-compatible point group."""
    target_indices = np.asarray(target_indices, dtype=np.int64)
    group_points = points[target_indices]
    group_selected = np.asarray(
        sorted(selected.intersection(target_indices.tolist())), dtype=np.int64
    )
    if group_selected.size == 0:
        order = np.lexsort(
            (
                target_indices,
                group_points[:, 2],
                group_points[:, 1],
                group_points[:, 0],
            )
        )
        first = int(target_indices[order[0]])
        selected.add(first)
        group_selected = np.array([first], dtype=np.int64)

    nearest = np.min(
        np.linalg.norm(
            group_points[:, None, :] - points[group_selected][None, :, :], axis=2
        ),
        axis=1,
    )
    tolerance = 1e-12 * max(1.0, max_distance)
    while float(np.max(nearest)) > max_distance + tolerance:
        farthest_distance = float(np.max(nearest))
        candidates = np.flatnonzero(
            np.isclose(nearest, farthest_distance, rtol=0.0, atol=tolerance)
        )
        next_index = int(
            target_indices[candidates[np.argmin(target_indices[candidates])]]
        )
        selected.add(next_index)
        distance = np.linalg.norm(group_points - points[next_index], axis=1)
        nearest = np.minimum(nearest, distance)


def _quadratic_endpoint_time_correction(
    sensitivity: RaySensitivity,
    anchor_source: np.ndarray,
    anchor_receiver: np.ndarray,
    delta_source: np.ndarray,
    delta_receiver: np.ndarray,
) -> float:
    r"""Return the fixed-topology second-order endpoint correction.

    For horizontal anchor offset :math:`R`, ray parameter :math:`p`, and
    :math:`X_p=\partial X/\partial p`, the directional curvature is

    .. math::

       T'' = X_p (p')^2 + p\|\delta r_\perp\|^2/R.

    The zero-offset limit replaces the second term by
    :math:`\|\delta r\|^2/X_p`.
    """
    x_p = float(sensitivity.doffset_dray_parameter)
    p = float(sensitivity.ray_parameter)
    if not np.isfinite(x_p) or x_p <= 0.0 or not np.isfinite(p):
        return 0.0

    delta_parameter = float(
        np.dot(sensitivity.dray_parameter_dsource, delta_source)
        + np.dot(sensitivity.dray_parameter_dreceiver, delta_receiver)
    )
    relative_change = delta_receiver[:2] - delta_source[:2]
    horizontal = anchor_receiver[:2] - anchor_source[:2]
    offset = float(np.linalg.norm(horizontal))
    curvature = x_p * delta_parameter**2
    if offset > 1e-12:
        direction = horizontal / offset
        transverse = relative_change - direction * np.dot(direction, relative_change)
        curvature += p * float(np.dot(transverse, transverse)) / offset
    else:
        curvature += float(np.dot(relative_change, relative_change)) / x_p
    return 0.5 * curvature


@dataclass
class TravelTimeApproximator:
    """Fitted topology-safe endpoint-Taylor traveltime approximator.

    Use :meth:`fit` to construct instances. The fitted velocity model and
    itinerary are fixed; model changes require refitting.

    Attributes
    ----------
    source_anchor_indices, receiver_anchor_indices : numpy.ndarray
        Original endpoint indices retained as anchors.
    anchor_sources, anchor_receivers : numpy.ndarray
        Anchor coordinates.
    anchor_trace : TraceResult
        Exact Cartesian-product ray trace at the anchors.
    source_max_distance, receiver_max_distance : float or None
        Endpoint coverage limits used during fitting.
    """

    velocity_model: pd.DataFrame
    source_phase: str
    itinerary: RayItinerary | None
    fit_sources: np.ndarray
    fit_receivers: np.ndarray
    source_max_distance: float | None
    receiver_max_distance: float | None
    source_anchor_indices: np.ndarray
    receiver_anchor_indices: np.ndarray
    anchor_sources: np.ndarray
    anchor_receivers: np.ndarray
    anchor_trace: TraceResult
    anchor_topologies: np.ndarray

    @property
    def exact_ray_count(self) -> int:
        """Number of exact source-anchor to receiver-anchor rays."""
        return len(self.anchor_sources) * len(self.anchor_receivers)

    @classmethod
    def fit(
        cls,
        sources,
        receivers,
        velocity_model: pd.DataFrame,
        *,
        source_max_distance: float | None,
        receiver_max_distance: float | None,
        source_phase: str = "P",
        itinerary: RayItinerary | None = None,
        n_jobs: int = -1,
        backend: str = "loky",
        sequential_limit: int = 5000,
        rays_per_chunk: int | None = None,
        tol: float = 1e-6,
        max_iter: int = 20,
        verbose: bool = False,
    ) -> "TravelTimeApproximator":
        """Fit exact anchors to a source-receiver target domain.

        Parameters
        ----------
        sources, receivers : array-like
            Endpoint coordinates with shape ``(N, 3)`` or ``(3,)``.
        velocity_model : pandas.DataFrame
            Layer model accepted by :func:`trace_rays`.
        source_max_distance, receiver_max_distance : float or None
            Maximum endpoint-to-anchor distances (m). ``None`` retains every
            supplied endpoint of that type as an exact anchor.
        source_phase : {"P", "SV", "SH", "S"}
            One direct-ray source phase. Ignored when ``itinerary`` is given.
        itinerary : RayItinerary, optional
            Explicit ordered fixed-topology path.
        n_jobs, backend, sequential_limit, rays_per_chunk, tol, max_iter, verbose
            Controls forwarded to the exact anchor call to :func:`trace_rays`.

        Returns
        -------
        TravelTimeApproximator
            Fitted anchor set and exact anchor rays.
        """
        source_points = _normalize_points(sources, "sources")
        receiver_points = _normalize_points(receivers, "receivers")
        if not isinstance(velocity_model, pd.DataFrame):
            raise TypeError("velocity_model must be a pandas DataFrame.")
        model_frame = velocity_model.copy(deep=True)
        model = ModelArrays.from_dataframe(model_frame)
        if itinerary is not None:
            if not isinstance(itinerary, RayItinerary):
                raise TypeError("itinerary must be a RayItinerary object.")
            phase = itinerary.source_phase
        else:
            if not isinstance(source_phase, str):
                raise ValueError("TravelTimeApproximator supports one source phase.")
            phase = normalize_phase(source_phase)

        for value, name in (
            (source_max_distance, "source_max_distance"),
            (receiver_max_distance, "receiver_max_distance"),
        ):
            if value is not None and (not np.isfinite(value) or float(value) <= 0.0):
                raise ValueError(f"{name} must be positive and finite, or None.")
        source_limit = None if source_max_distance is None else float(source_max_distance)
        receiver_limit = None if receiver_max_distance is None else float(receiver_max_distance)

        source_layers = _layer_indices(source_points[:, 2], model.depths)
        receiver_layers = _layer_indices(receiver_points[:, 2], model.depths)
        if source_limit is None:
            source_indices = np.arange(len(source_points), dtype=np.int64)
        else:
            source_indices = select_anchors(source_points, source_limit, source_layers).indices
        if receiver_limit is None:
            receiver_indices = np.arange(len(receiver_points), dtype=np.int64)
        else:
            receiver_indices = select_anchors(receiver_points, receiver_limit, receiver_layers).indices

        topology_cache: dict[tuple[float, float, bool], object] = {}

        def topology(source_point, receiver_point):
            horizontal = float(np.linalg.norm(receiver_point[:2] - source_point[:2]))
            cache_key = (
                float(source_point[2]),
                float(receiver_point[2]),
                horizontal <= 1e-9,
            )
            if cache_key not in topology_cache:
                topology_cache[cache_key] = _topology_key(
                    model, source_point, receiver_point, phase, itinerary
                )
            return topology_cache[cache_key]

        target_topologies = np.empty(len(source_points) * len(receiver_points), dtype=object)
        for source_index, source_point in enumerate(source_points):
            for receiver_index, receiver_point in enumerate(receiver_points):
                flat_index = source_index * len(receiver_points) + receiver_index
                target_topologies[flat_index] = topology(source_point, receiver_point)

        source_set = set(source_indices.tolist())
        receiver_set = set(receiver_indices.tolist())
        if receiver_limit is None and source_limit is not None:
            for receiver_index in range(len(receiver_points)):
                grouped_sources: dict[object, list[int]] = {}
                for source_index in range(len(source_points)):
                    key = target_topologies[
                        source_index * len(receiver_points) + receiver_index
                    ]
                    if key is not None:
                        grouped_sources.setdefault(key, []).append(source_index)
                for group_indices in grouped_sources.values():
                    _extend_anchor_cover(
                        source_points,
                        np.asarray(group_indices, dtype=np.int64),
                        source_set,
                        source_limit,
                    )
        elif source_limit is None and receiver_limit is not None:
            for source_index in range(len(source_points)):
                grouped_receivers: dict[object, list[int]] = {}
                start = source_index * len(receiver_points)
                for receiver_index in range(len(receiver_points)):
                    key = target_topologies[start + receiver_index]
                    if key is not None:
                        grouped_receivers.setdefault(key, []).append(receiver_index)
                for group_indices in grouped_receivers.values():
                    _extend_anchor_cover(
                        receiver_points,
                        np.asarray(group_indices, dtype=np.int64),
                        receiver_set,
                        receiver_limit,
                    )
        else:
            while True:
                sorted_sources = np.asarray(sorted(source_set), dtype=np.int64)
                sorted_receivers = np.asarray(sorted(receiver_set), dtype=np.int64)
                pair_source_indices = np.repeat(sorted_sources, len(sorted_receivers))
                pair_receiver_indices = np.tile(sorted_receivers, len(sorted_sources))
                anchor_groups: dict[object, list[int]] = {}
                for pair_index, (source_index, receiver_index) in enumerate(zip(
                    pair_source_indices, pair_receiver_indices
                )):
                    key = topology(source_points[source_index], receiver_points[receiver_index])
                    if key is not None:
                        anchor_groups.setdefault(key, []).append(pair_index)

                uncovered_groups: dict[object, list[tuple[int, int]]] = {}
                for flat_index, target_key in enumerate(target_topologies):
                    if target_key is None:
                        continue
                    source_index = flat_index // len(receiver_points)
                    receiver_index = flat_index % len(receiver_points)
                    candidate_positions = anchor_groups.get(target_key, ())
                    if candidate_positions:
                        candidate_positions = np.asarray(candidate_positions, dtype=np.int64)
                        source_distances = np.linalg.norm(
                            source_points[pair_source_indices[candidate_positions]]
                            - source_points[source_index],
                            axis=1,
                        )
                        receiver_distances = np.linalg.norm(
                            receiver_points[pair_receiver_indices[candidate_positions]]
                            - receiver_points[receiver_index],
                            axis=1,
                        )
                        covered = np.any(
                            _distance_allowed(source_distances, source_limit)
                            & _distance_allowed(receiver_distances, receiver_limit)
                        )
                    else:
                        covered = False
                    if not covered:
                        uncovered_groups.setdefault(target_key, []).append(
                            (source_index, receiver_index)
                        )

                if not uncovered_groups:
                    break

                previous_sizes = (len(source_set), len(receiver_set))
                for uncovered in uncovered_groups.values():
                    uncovered_array = np.asarray(uncovered, dtype=np.int64)
                    candidate_sources = np.unique(uncovered_array[:, 0])
                    candidate_receivers = np.unique(uncovered_array[:, 1])
                    if source_limit is None:
                        source_set.update(candidate_sources.tolist())
                    else:
                        selected = select_anchors(
                            source_points[candidate_sources], source_limit
                        ).indices
                        source_set.update(candidate_sources[selected].tolist())
                    if receiver_limit is None:
                        receiver_set.update(candidate_receivers.tolist())
                    else:
                        selected = select_anchors(
                            receiver_points[candidate_receivers], receiver_limit
                        ).indices
                        receiver_set.update(candidate_receivers[selected].tolist())
                if previous_sizes == (len(source_set), len(receiver_set)):
                    first_uncovered = next(iter(uncovered_groups.values()))[0]
                    source_set.add(first_uncovered[0])
                    receiver_set.add(first_uncovered[1])

        source_indices = np.asarray(sorted(source_set), dtype=np.int64)
        receiver_indices = np.asarray(sorted(receiver_set), dtype=np.int64)
        anchor_sources = source_points[source_indices]
        anchor_receivers = receiver_points[receiver_indices]
        anchor_trace = trace_rays(
            anchor_sources,
            anchor_receivers,
            model_frame,
            source_phase=phase,
            itinerary=itinerary,
            requested={"travel_times", "sensitivities", "diagnostics"},
            n_jobs=n_jobs,
            backend=backend,
            sequential_limit=sequential_limit,
            rays_per_chunk=rays_per_chunk,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
        )
        anchor_topologies = np.empty(len(anchor_sources) * len(anchor_receivers), dtype=object)
        for source_index, source_point in enumerate(anchor_sources):
            for receiver_index, receiver_point in enumerate(anchor_receivers):
                flat_index = source_index * len(anchor_receivers) + receiver_index
                anchor_topologies[flat_index] = topology(source_point, receiver_point)

        return cls(
            velocity_model=model_frame,
            source_phase=phase,
            itinerary=itinerary,
            fit_sources=source_points.copy(),
            fit_receivers=receiver_points.copy(),
            source_max_distance=source_limit,
            receiver_max_distance=receiver_limit,
            source_anchor_indices=source_indices,
            receiver_anchor_indices=receiver_indices,
            anchor_sources=anchor_sources.copy(),
            anchor_receivers=anchor_receivers.copy(),
            anchor_trace=anchor_trace,
            anchor_topologies=anchor_topologies,
        )

    def predict(self, sources=None, receivers=None, *, order: int = 1) -> TravelTimePrediction:
        """Predict traveltimes for targets inside the fitted anchor domain.

        Parameters
        ----------
        sources, receivers : array-like, optional
            Query endpoints. Omitted arrays default to the corresponding
            points supplied to :meth:`fit`.
        order : {1, 2}
            Endpoint Taylor order. Second order adds fixed-topology curvature
            and evaluates direct same-layer paths exactly. The default retains
            the original first-order predictor.

        Returns
        -------
        TravelTimePrediction
            Source-major predictions, masks, reason codes, and anchor mapping.

        Notes
        -----
        Unsupported targets are returned as ``NaN`` and are never retraced.
        Possible reason codes are ``"unreachable_topology"``,
        ``"no_topology_match"``, ``"outside_anchor_distance"``, and
        ``"invalid_anchor_sensitivity"``.
        """
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2.")
        source_points = self.fit_sources if sources is None else _normalize_points(sources, "sources")
        receiver_points = self.fit_receivers if receivers is None else _normalize_points(receivers, "receivers")
        model = ModelArrays.from_dataframe(self.velocity_model)
        n_rays = len(source_points) * len(receiver_points)
        travel_times = np.full(n_rays, np.nan, dtype=np.float64)
        valid = np.zeros(n_rays, dtype=bool)
        reasons = np.full(n_rays, _REASON_NO_MATCH, dtype="<U32")
        anchor_indices = np.full(n_rays, -1, dtype=np.int64)
        source_used_distances = np.full(n_rays, np.nan, dtype=np.float64)
        receiver_used_distances = np.full(n_rays, np.nan, dtype=np.float64)
        anchor_groups: dict[object, np.ndarray] = {}
        grouped_positions: dict[object, list[int]] = {}
        for index, key in enumerate(self.anchor_topologies):
            if key is not None:
                grouped_positions.setdefault(key, []).append(index)
        for key, positions in grouped_positions.items():
            anchor_groups[key] = np.asarray(positions, dtype=np.int64)
        sensitivity_valid = np.asarray(
            [
                sensitivity is not None and sensitivity.valid
                for sensitivity in self.anchor_trace.sensitivities
            ],
            dtype=bool,
        )
        receiver_distance_matrix = np.linalg.norm(
            receiver_points[:, None, :] - self.anchor_receivers[None, :, :], axis=2
        )
        source_layers = _layer_indices(source_points[:, 2], model.depths)
        receiver_layers = _layer_indices(receiver_points[:, 2], model.depths)
        topology_cache: dict[tuple[float, float, bool], object] = {}

        for source_index, source_point in enumerate(source_points):
            source_distances = np.linalg.norm(self.anchor_sources - source_point, axis=1)
            for receiver_index, receiver_point in enumerate(receiver_points):
                flat_index = source_index * len(receiver_points) + receiver_index
                horizontal = float(np.linalg.norm(receiver_point[:2] - source_point[:2]))
                cache_key = (
                    float(source_point[2]),
                    float(receiver_point[2]),
                    horizontal <= 1e-9,
                )
                if cache_key not in topology_cache:
                    topology_cache[cache_key] = _topology_key(
                        model,
                        source_point,
                        receiver_point,
                        self.source_phase,
                        self.itinerary,
                    )
                target_key = topology_cache[cache_key]
                if target_key is None:
                    reasons[flat_index] = _REASON_UNREACHABLE
                    continue
                topology_candidates = anchor_groups.get(target_key, np.empty(0, dtype=np.int64))
                if topology_candidates.size == 0:
                    reasons[flat_index] = _REASON_NO_MATCH
                    continue

                receiver_distances = receiver_distance_matrix[receiver_index]
                candidate_source_indices = topology_candidates // len(self.anchor_receivers)
                candidate_receiver_indices = topology_candidates % len(self.anchor_receivers)
                candidate_source_distances = source_distances[candidate_source_indices]
                candidate_receiver_distances = receiver_distances[candidate_receiver_indices]
                within_distance = (
                    _distance_allowed(candidate_source_distances, self.source_max_distance)
                    & _distance_allowed(candidate_receiver_distances, self.receiver_max_distance)
                )
                if not np.any(within_distance):
                    reasons[flat_index] = _REASON_OUTSIDE
                    continue
                candidates = topology_candidates[within_distance]
                candidate_source_distances = candidate_source_distances[within_distance]
                candidate_receiver_distances = candidate_receiver_distances[within_distance]
                ranking = np.hypot(
                    _normalized_distance(candidate_source_distances, self.source_max_distance),
                    _normalized_distance(candidate_receiver_distances, self.receiver_max_distance),
                )
                ranking[~sensitivity_valid[candidates]] = np.inf
                position = int(np.argmin(ranking))
                if not np.isfinite(ranking[position]):
                    reasons[flat_index] = _REASON_INVALID_SENSITIVITY
                    continue

                candidate = int(candidates[position])
                sensitivity = self.anchor_trace.sensitivities[candidate]
                anchor_source_index = candidate // len(self.anchor_receivers)
                anchor_receiver_index = candidate % len(self.anchor_receivers)
                anchor_source = self.anchor_sources[anchor_source_index]
                anchor_receiver = self.anchor_receivers[anchor_receiver_index]
                delta_source = source_point - anchor_source
                delta_receiver = receiver_point - anchor_receiver
                delta_time = float(
                    np.dot(sensitivity.dtravel_time_dsource, delta_source)
                    + np.dot(sensitivity.dtravel_time_dreceiver, delta_receiver)
                )
                predicted_time = self.anchor_trace.travel_times[candidate] + delta_time
                if order == 2:
                    if (
                        self.itinerary is None
                        and source_layers[source_index] == receiver_layers[receiver_index]
                    ):
                        layer_index = source_layers[source_index]
                        velocity = (
                            model.vp[layer_index]
                            if self.source_phase == "P"
                            else model.vs[layer_index]
                        )
                        predicted_time = float(
                            np.linalg.norm(receiver_point - source_point) / velocity
                        )
                    else:
                        predicted_time += _quadratic_endpoint_time_correction(
                            sensitivity,
                            anchor_source,
                            anchor_receiver,
                            delta_source,
                            delta_receiver,
                        )
                travel_times[flat_index] = predicted_time
                valid[flat_index] = True
                reasons[flat_index] = _REASON_OK
                anchor_indices[flat_index] = candidate
                source_used_distances[flat_index] = candidate_source_distances[position]
                receiver_used_distances[flat_index] = candidate_receiver_distances[position]

        return TravelTimePrediction(
            travel_times=travel_times,
            valid_mask=valid,
            reasons=reasons,
            anchor_ray_indices=anchor_indices,
            source_anchor_distances=source_used_distances,
            receiver_anchor_distances=receiver_used_distances,
            n_sources=len(source_points),
            n_receivers=len(receiver_points),
        )
