r"""Analytic fixed-topology traveltime sensitivities.

The routines in this module differentiate a solved, prescribed ray path.
They do not select ray branches or differentiate changes in path topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import ModelArrays, normalize_phase


@dataclass(frozen=True)
class PathLeg:
    """One atomic traversal of a physical model layer."""

    layer_index: int
    phase: str
    z_start: float
    z_end: float
    thickness: float
    velocity: float

    @property
    def direction(self) -> float:
        """Return ``+1`` downward and ``-1`` upward."""
        return 1.0 if self.z_end > self.z_start else -1.0


@dataclass
class RaySensitivity:
    r"""Sparse derivatives for one fixed-topology ray.

    Layer derivative arrays contain values only for the model-layer indices
    listed by the corresponding ``*_layer_indices`` field. Interface indices
    are row indices into the model ``Depth`` array; the surface row is omitted.

    The derivatives are valid while layer membership, prescribed phase
    itinerary, and the propagating branch remain unchanged.

    ``ray_parameter`` and ``doffset_dray_parameter`` retain :math:`p` and
    :math:`X_p=\partial X/\partial p` from the solved ray.  They support the
    optional second-order endpoint predictor without another ray solve.
    """

    vp_layer_indices: np.ndarray
    dtravel_time_dvp: np.ndarray
    dray_parameter_dvp: np.ndarray
    vs_layer_indices: np.ndarray
    dtravel_time_dvs: np.ndarray
    dray_parameter_dvs: np.ndarray
    interface_indices: np.ndarray
    dtravel_time_dinterface_depths: np.ndarray
    dray_parameter_dinterface_depths: np.ndarray
    dtravel_time_dsource: np.ndarray
    dtravel_time_dreceiver: np.ndarray
    dray_parameter_dsource: np.ndarray
    dray_parameter_dreceiver: np.ndarray
    ray_parameter: float = np.nan
    doffset_dray_parameter: float = np.nan
    valid: bool = True
    reason: str | None = None


def _empty_sensitivity(valid: bool, reason: str | None = None) -> RaySensitivity:
    """Return an empty sensitivity record."""
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=np.float64)
    endpoint = np.zeros(3, dtype=np.float64) if valid else np.full(3, np.nan)
    return RaySensitivity(
        vp_layer_indices=empty_i.copy(),
        dtravel_time_dvp=empty_f.copy(),
        dray_parameter_dvp=empty_f.copy(),
        vs_layer_indices=empty_i.copy(),
        dtravel_time_dvs=empty_f.copy(),
        dray_parameter_dvs=empty_f.copy(),
        interface_indices=empty_i.copy(),
        dtravel_time_dinterface_depths=empty_f.copy(),
        dray_parameter_dinterface_depths=empty_f.copy(),
        dtravel_time_dsource=endpoint.copy(),
        dtravel_time_dreceiver=endpoint.copy(),
        dray_parameter_dsource=endpoint.copy(),
        dray_parameter_dreceiver=endpoint.copy(),
        ray_parameter=np.nan,
        doffset_dray_parameter=np.nan,
        valid=valid,
        reason=reason,
    )


def _layer_index(depth: float, boundaries: np.ndarray) -> int:
    """Return the physical model-layer index at an interior point."""
    return max(int(np.searchsorted(boundaries, depth, side="right")) - 1, 0)


def compile_path_legs(
    model: ModelArrays,
    segments: Sequence[dict],
) -> list[PathLeg]:
    """Split monotonic phase segments into physical-layer traversals."""
    legs: list[PathLeg] = []
    boundaries = model.depths

    for segment in segments:
        z_start = float(segment["start_z"])
        z_end = float(segment["end_z"])
        if np.isclose(z_start, z_end, rtol=0.0, atol=1e-12):
            continue

        phase = normalize_phase(segment["phase"])
        z_lo, z_hi = sorted((z_start, z_end))
        internal = boundaries[(boundaries > z_lo) & (boundaries < z_hi)]
        if z_end > z_start:
            points = np.concatenate(([z_start], internal, [z_end]))
        else:
            points = np.concatenate(([z_start], internal[::-1], [z_end]))

        for leg_start, leg_end in zip(points[:-1], points[1:]):
            midpoint = 0.5 * (leg_start + leg_end)
            layer_index = _layer_index(midpoint, boundaries)
            velocity = (
                model.vp[layer_index] if phase == "P" else model.vs[layer_index]
            )
            legs.append(
                PathLeg(
                    layer_index=layer_index,
                    phase=phase,
                    z_start=float(leg_start),
                    z_end=float(leg_end),
                    thickness=float(abs(leg_end - leg_start)),
                    velocity=float(velocity),
                )
            )

    return legs


def _group_layer_derivatives(
    layer_indices: np.ndarray,
    mask: np.ndarray,
    dtime: np.ndarray,
    dparameter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum per-leg derivatives into sparse physical-layer arrays."""
    selected = layer_indices[mask]
    if selected.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    unique, inverse = np.unique(selected, return_inverse=True)
    time_grouped = np.zeros(unique.size, dtype=np.float64)
    parameter_grouped = np.zeros(unique.size, dtype=np.float64)
    np.add.at(time_grouped, inverse, dtime[mask])
    np.add.at(parameter_grouped, inverse, dparameter[mask])
    return unique.astype(np.int64), time_grouped, parameter_grouped


def _horizontal_path_sensitivity(
    model: ModelArrays,
    source: np.ndarray,
    receiver: np.ndarray,
    source_phase: str,
) -> RaySensitivity:
    """Return the same-layer horizontal-ray limit."""
    delta = np.asarray(receiver, dtype=float) - np.asarray(source, dtype=float)
    horizontal = float(np.hypot(delta[0], delta[1]))
    if horizontal <= 1e-12:
        return _empty_sensitivity(False, "Sensitivity is undefined for coincident endpoints.")

    phase = normalize_phase(source_phase)
    layer_index = _layer_index(float(source[2]), model.depths)
    velocity = float(model.vp[layer_index] if phase == "P" else model.vs[layer_index])
    unit = delta[:2] / horizontal
    dtime_source = np.array([-unit[0] / velocity, -unit[1] / velocity, 0.0])
    dtime_receiver = -dtime_source

    kwargs = {
        "vp_layer_indices": np.empty(0, dtype=np.int64),
        "dtravel_time_dvp": np.empty(0),
        "dray_parameter_dvp": np.empty(0),
        "vs_layer_indices": np.empty(0, dtype=np.int64),
        "dtravel_time_dvs": np.empty(0),
        "dray_parameter_dvs": np.empty(0),
    }
    prefix = "vp" if phase == "P" else "vs"
    kwargs[f"{prefix}_layer_indices"] = np.array([layer_index], dtype=np.int64)
    kwargs[f"dtravel_time_d{prefix}"] = np.array([-horizontal / velocity**2])
    kwargs[f"dray_parameter_d{prefix}"] = np.array([-1.0 / velocity**2])

    return RaySensitivity(
        **kwargs,
        interface_indices=np.empty(0, dtype=np.int64),
        dtravel_time_dinterface_depths=np.empty(0),
        dray_parameter_dinterface_depths=np.empty(0),
        dtravel_time_dsource=dtime_source,
        dtravel_time_dreceiver=dtime_receiver,
        dray_parameter_dsource=np.zeros(3),
        dray_parameter_dreceiver=np.zeros(3),
        ray_parameter=1.0 / velocity,
        doffset_dray_parameter=np.inf,
    )


def compute_ray_sensitivity(
    model: ModelArrays,
    source: np.ndarray,
    receiver: np.ndarray,
    source_phase: str,
    segments: Sequence[dict],
    ray_parameter: float | None,
) -> RaySensitivity:
    r"""Differentiate one solved ray at fixed horizontal offset.

    For leg :math:`j`, let :math:`c_j=\sqrt{1-p^2v_j^2}`. The offset
    derivative is

    .. math::

       X_p = \sum_j h_j v_j / c_j^3.

    Implicit differentiation gives :math:`p_m=-X_m/X_p`, while Fermat
    stationarity reduces the fixed-offset velocity and thickness derivatives
    to :math:`T_{v_j}=-h_j/(v_j^2c_j)` and
    :math:`T_{h_j}=c_j/v_j`, respectively.
    """
    source = np.asarray(source, dtype=np.float64)
    receiver = np.asarray(receiver, dtype=np.float64)
    legs = compile_path_legs(model, segments)
    if not legs:
        return _horizontal_path_sensitivity(model, source, receiver, source_phase)
    if ray_parameter is None or not np.isfinite(ray_parameter):
        return _empty_sensitivity(False, "The solved ray parameter is not finite.")

    p = float(ray_parameter)
    h = np.array([leg.thickness for leg in legs], dtype=np.float64)
    v = np.array([leg.velocity for leg in legs], dtype=np.float64)
    pv = p * v
    if np.any(np.abs(pv) >= 1.0):
        return _empty_sensitivity(False, "The ray is critical or evanescent.")

    c = np.sqrt(1.0 - pv * pv)
    x_p = float(np.sum(h * v / c**3))
    if not np.isfinite(x_p) or x_p <= 0.0:
        return _empty_sensitivity(False, "The offset derivative is not positive and finite.")

    layer_indices = np.array([leg.layer_index for leg in legs], dtype=np.int64)
    phases = np.array([leg.phase for leg in legs])
    dtime_dvelocity = -h / (v**2 * c)
    dparameter_dvelocity = -(h * p / c**3) / x_p

    vp_indices, dtime_dvp, dparameter_dvp = _group_layer_derivatives(
        layer_indices, phases == "P", dtime_dvelocity, dparameter_dvelocity
    )
    vs_indices, dtime_dvs, dparameter_dvs = _group_layer_derivatives(
        layer_indices, phases != "P", dtime_dvelocity, dparameter_dvelocity
    )

    interface_indices: list[int] = []
    dtime_interfaces: list[float] = []
    dparameter_interfaces: list[float] = []
    depth_tolerance = 1e-8
    for interface_index, depth in enumerate(model.depths[1:], start=1):
        thickness_coeff = np.zeros(len(legs), dtype=np.float64)
        for index, leg in enumerate(legs):
            if abs(leg.z_end - depth) <= depth_tolerance:
                thickness_coeff[index] += leg.direction
            if abs(leg.z_start - depth) <= depth_tolerance:
                thickness_coeff[index] -= leg.direction
        if np.any(thickness_coeff):
            interface_indices.append(interface_index)
            dtime_interfaces.append(float(np.sum(thickness_coeff * c / v)))
            x_interface = float(np.sum(thickness_coeff * p * v / c))
            dparameter_interfaces.append(-x_interface / x_p)

    delta = receiver - source
    horizontal = float(np.hypot(delta[0], delta[1]))
    if horizontal > 1e-12:
        ux, uy = delta[0] / horizontal, delta[1] / horizontal
    else:
        ux, uy = 0.0, 0.0

    first_direction = legs[0].direction
    last_direction = legs[-1].direction
    source_h_coeff = -first_direction
    receiver_h_coeff = last_direction
    source_eta = c[0] / v[0]
    receiver_eta = c[-1] / v[-1]
    source_x_h = source_h_coeff * p * v[0] / c[0]
    receiver_x_h = receiver_h_coeff * p * v[-1] / c[-1]

    dtime_source = np.array([-p * ux, -p * uy, source_h_coeff * source_eta])
    dtime_receiver = np.array([p * ux, p * uy, receiver_h_coeff * receiver_eta])
    dparameter_source = np.array([-ux / x_p, -uy / x_p, -source_x_h / x_p])
    dparameter_receiver = np.array([ux / x_p, uy / x_p, -receiver_x_h / x_p])

    return RaySensitivity(
        vp_layer_indices=vp_indices,
        dtravel_time_dvp=dtime_dvp,
        dray_parameter_dvp=dparameter_dvp,
        vs_layer_indices=vs_indices,
        dtravel_time_dvs=dtime_dvs,
        dray_parameter_dvs=dparameter_dvs,
        interface_indices=np.asarray(interface_indices, dtype=np.int64),
        dtravel_time_dinterface_depths=np.asarray(dtime_interfaces, dtype=np.float64),
        dray_parameter_dinterface_depths=np.asarray(dparameter_interfaces, dtype=np.float64),
        dtravel_time_dsource=dtime_source,
        dtravel_time_dreceiver=dtime_receiver,
        dray_parameter_dsource=dparameter_source,
        dray_parameter_dreceiver=dparameter_receiver,
        ray_parameter=p,
        doffset_dray_parameter=x_p,
    )
