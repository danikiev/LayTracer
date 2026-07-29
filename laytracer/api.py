r"""
High-level multi-ray tracing interface.

Provides :func:`trace_rays`, the main entry point for tracing all
source-receiver pairs through a 1-D layered velocity model, with
optional parallel execution using the ``loky`` backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed

from .model import ModelArrays, build_layer_stack, normalize_phase, velocity_key
from .sensitivity import RaySensitivity, compute_ray_sensitivity
from .solver import SolveDiagnostics, solve, transmission_product


TRACE_OUTPUTS = frozenset({
    "travel_times",
    "rays",
    "ray_parameters",
    "tstar",
    "spreading",
    "trans_product",
    "complex_coefficient_product",
    "diagnostics",
    "sensitivities",
})
DEFAULT_REQUESTED = ("travel_times", "rays", "ray_parameters")


@dataclass(frozen=True)
class Interaction:
    """One ordered reflection or transmission in a prescribed itinerary.

    Parameters
    ----------
    depth : float
        Model-interface depth (m, positive downward).
    kind : {"reflect", "transmit"}
        Whether propagation reverses or retains its vertical direction.
    outgoing_phase : {"P", "SV", "SH", "S"}
        Phase immediately after the interaction. ``"S"`` is normalized to
        ``"SV"``.
    """

    depth: float
    kind: str
    outgoing_phase: str

    def __post_init__(self) -> None:
        depth = float(self.depth)
        kind = str(self.kind).lower()
        if not np.isfinite(depth):
            raise ValueError("Interaction depth must be finite.")
        if kind not in {"reflect", "transmit"}:
            raise ValueError("Interaction kind must be 'reflect' or 'transmit'.")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "outgoing_phase", normalize_phase(self.outgoing_phase))


@dataclass(frozen=True)
class RayItinerary:
    """An ordered, fixed-topology phase itinerary."""

    source_phase: str
    interactions: Sequence[Interaction]

    def __post_init__(self) -> None:
        normalized: list[Interaction] = []
        for interaction in self.interactions:
            if not isinstance(interaction, Interaction):
                raise TypeError("RayItinerary interactions must be Interaction objects.")
            normalized.append(interaction)
        object.__setattr__(self, "source_phase", normalize_phase(self.source_phase))
        object.__setattr__(self, "interactions", tuple(normalized))


@dataclass
class TraceResult:
    """Container for multi-ray tracing results.

    Attributes
    ----------
    travel_times : numpy.ndarray
        Travel times (s), shape ``(n_rays,)``.
    rays : list of numpy.ndarray or None
        Ray paths; each element is shape ``(M_i, 3)`` in the original
        3-D coordinate system.  *None* if not requested.
    ray_parameters : numpy.ndarray or None
        Horizontal slowness *p* for each ray, shape ``(n_rays,)``.
    tstar : numpy.ndarray or None
        Attenuation operator :math:`t^*` for each ray, shape ``(n_rays,)``.
    spreading : numpy.ndarray or None
        Relative geometrical spreading factor for each ray, shape ``(n_rays,)``.
    trans_product : numpy.ndarray or None
        Real, nonnegative product of interface-coefficient magnitudes along
        each ray.
    source_phase : str or None
        Canonical source phase for this result.
    complex_coefficient_product : numpy.ndarray or None
        Signed/complex cumulative interface coefficient for each ray.
    diagnostics : list of SolveDiagnostics or None
        Per-ray numerical certificates, if requested.
    sensitivities : list of RaySensitivity or None
        Per-ray fixed-topology analytic derivatives, if requested.
    """

    travel_times: np.ndarray
    rays: list[np.ndarray] | None = None
    ray_parameters: np.ndarray | None = None
    tstar: np.ndarray | None = None
    spreading: np.ndarray | None = None
    trans_product: np.ndarray | None = None
    source_phase: str | None = None
    complex_coefficient_product: np.ndarray | None = None
    diagnostics: list[SolveDiagnostics | None] | None = None
    sensitivities: list[RaySensitivity | None] | None = None


def _normalize_requested(requested: Sequence[str] | None) -> frozenset[str]:
    """Validate and normalize the requested output names."""
    if requested is None:
        requested = DEFAULT_REQUESTED

    normalized = frozenset(str(name) for name in requested)
    invalid = normalized - TRACE_OUTPUTS
    if invalid:
        valid = ", ".join(sorted(TRACE_OUTPUTS))
        invalid_str = ", ".join(sorted(invalid))
        raise ValueError(f"Invalid requested outputs: {invalid_str}. Valid outputs: {valid}")

    if "travel_times" not in normalized:
        raise ValueError("requested must include 'travel_times'")

    return normalized


def _normalize_source_phases(source_phase) -> tuple[tuple[str, ...], bool]:
    """Normalize source_phase while preserving list order and de-duplicating."""
    if isinstance(source_phase, str):
        return (normalize_phase(source_phase),), False

    phases: list[str] = []
    seen = set()
    for phase in source_phase:
        phase_norm = normalize_phase(phase)
        if phase_norm not in seen:
            phases.append(phase_norm)
            seen.add(phase_norm)

    if not phases:
        raise ValueError("source_phase must contain at least one phase.")

    return tuple(phases), True


def _normalize_interaction_phases(
    arg: Sequence[tuple[float, str]] | None,
) -> list[tuple[float, str]]:
    """Normalize interaction phase labels."""
    if arg is None:
        return []
    return [(float(z), normalize_phase(ph)) for z, ph in arg]


def _validate_sh_coupling(source_phase: str, interactions: list[tuple[float, str]]) -> None:
    """Reject SH/P-SV mode conversions in isotropic media."""
    source_is_sh = source_phase == "SH"
    for _, phase in interactions:
        phase_is_sh = phase == "SH"
        if source_is_sh != phase_is_sh:
            raise ValueError("SH is decoupled from P-SV in isotropic 1-D media.")


def _trace_batch(batch):
    """Worker function for parallel ray computation."""
    (
        batch_indices,
        source_coords,
        receiver_coords,
        model_arrays,
        source_phases,
        return_dict,
        refl_list,
        refr_list,
        ordered_interactions,
        need_rays,
        need_ray_parameters,
        need_tstar,
        need_spreading,
        need_trans_product,
        need_complex_coefficient_product,
        need_diagnostics,
        need_sensitivities,
        transcoef_method,
        tol,
        max_iter,
    ) = batch

    results = []
    for isrc, ircv in batch_indices:
        if return_dict:
            results.append(
                _trace_one_many(
                    ma=model_arrays,
                    src=source_coords[isrc],
                    rcv=receiver_coords[ircv],
                    source_phases=source_phases,
                    refl_list=refl_list,
                    refr_list=refr_list,
                    ordered_interactions=ordered_interactions,
                    need_rays=need_rays,
                    need_ray_parameters=need_ray_parameters,
                    need_tstar=need_tstar,
                    need_spreading=need_spreading,
                    need_trans_product=need_trans_product,
                    need_complex_coefficient_product=need_complex_coefficient_product,
                    need_diagnostics=need_diagnostics,
                    need_sensitivities=need_sensitivities,
                    transcoef_method=transcoef_method,
                    tol=tol,
                    max_iter=max_iter,
                )
            )
        else:
            results.append(
                _trace_one(
                    ma=model_arrays,
                    src=source_coords[isrc],
                    rcv=receiver_coords[ircv],
                    source_phase=source_phases[0],
                    refl_list=refl_list,
                    refr_list=refr_list,
                    ordered_interactions=ordered_interactions,
                    need_rays=need_rays,
                    need_ray_parameters=need_ray_parameters,
                    need_tstar=need_tstar,
                    need_spreading=need_spreading,
                    need_trans_product=need_trans_product,
                    need_complex_coefficient_product=need_complex_coefficient_product,
                    need_diagnostics=need_diagnostics,
                    need_sensitivities=need_sensitivities,
                    transcoef_method=transcoef_method,
                    tol=tol,
                    max_iter=max_iter,
                )
            )
    return results


def _project_ray_to_3d(
    ray2d: np.ndarray | None,
    sx: float,
    sy: float,
    dx: float,
    dy: float,
    epic: float,
) -> np.ndarray | None:
    """Project a 2-D ray in the epicentral plane back into 3-D."""
    if ray2d is None:
        return None

    mpts = ray2d.shape[0]
    ray3d = np.empty((mpts, 3))
    if epic > 1e-10:
        ux, uy = dx / epic, dy / epic
    else:
        ux, uy = 1.0, 0.0
    ray3d[:, 0] = sx + ray2d[:, 0] * ux
    ray3d[:, 1] = sy + ray2d[:, 0] * uy
    ray3d[:, 2] = ray2d[:, 1]
    return ray3d


def _direct_segments_for_phase(
    ma: ModelArrays,
    z_src: float,
    z_rcv: float,
    source_phase: str,
) -> list[dict] | None:
    """Build direct-path segments for coefficient reuse."""
    stack = build_layer_stack(ma, z_src, z_rcv)
    vel = stack.v(velocity_key(source_phase))
    valid = stack.h > 1e-9
    if not np.any(valid):
        return None

    return [{
        "h": stack.h[valid],
        "v": vel[valid],
        "vp": stack.vp[valid],
        "vs": stack.vs[valid],
        "rho": stack.rho[valid] if stack.rho is not None else None,
        "qp": stack.qp[valid] if stack.qp is not None else None,
        "qs": stack.qs[valid] if stack.qs is not None else None,
        "phase": source_phase,
        "start_z": z_src,
        "end_z": z_rcv,
    }]


def _direct_trans_product(
    ma: ModelArrays,
    z_src: float,
    z_rcv: float,
    source_phase: str,
    ray_parameter: float | None,
    transcoef_method: str,
) -> float:
    """Compute direct-path transmission product for a solved ray."""
    segments = _direct_segments_for_phase(ma, z_src, z_rcv, source_phase)
    if segments is None:
        return 1.0
    p = 0.0 if ray_parameter is None or not np.isfinite(ray_parameter) else float(ray_parameter)
    return transmission_product(p, segments, [], transcoef_method)


def _segment_between(
    ma: ModelArrays,
    start_z: float,
    end_z: float,
    phase: str,
) -> dict | None:
    """Build one monotonic solver segment."""
    stack = build_layer_stack(ma, start_z, end_z)
    valid = stack.h > 1e-9
    if not np.any(valid):
        return None
    velocity = stack.v(velocity_key(phase))
    return {
        "h": stack.h[valid],
        "v": velocity[valid],
        "vp": stack.vp[valid],
        "vs": stack.vs[valid],
        "rho": stack.rho[valid] if stack.rho is not None else None,
        "qp": stack.qp[valid] if stack.qp is not None else None,
        "qs": stack.qs[valid] if stack.qs is not None else None,
        "phase": phase,
        "start_z": start_z,
        "end_z": end_z,
    }


def _material_beyond_interaction(
    ma: ModelArrays,
    depth: float,
    going_down: bool,
) -> dict[str, float]:
    """Return material properties immediately beyond an interaction."""
    delta = 1.0
    if going_down:
        stack = build_layer_stack(ma, depth, depth + delta)
    else:
        stack = build_layer_stack(ma, depth - delta, depth)
    return {
        "vp": float(stack.vp[0]),
        "vs": float(stack.vs[0]),
        "rho": float(stack.rho[0]) if stack.rho is not None else 0.0,
    }


def _compile_ordered_itinerary(
    ma: ModelArrays,
    source_depth: float,
    receiver_depth: float,
    source_phase: str,
    interactions: tuple[Interaction, ...],
) -> tuple[list[dict], list[dict]]:
    """Compile an explicit ordered itinerary into solver metadata."""
    ray_segments: list[dict] = []
    interaction_metadata: list[dict] = []
    current_depth = source_depth
    current_phase = source_phase
    direction: float | None = None
    depth_tolerance = 1e-9

    for interaction in interactions:
        delta = interaction.depth - current_depth
        if abs(delta) <= depth_tolerance:
            raise ValueError(
                f"Cannot {interaction.kind} at the starting depth {interaction.depth} immediately."
            )
        incoming_direction = 1.0 if delta > 0.0 else -1.0
        if direction is not None and incoming_direction != direction:
            raise ValueError(
                f"Interaction at depth {interaction.depth} is not reachable in the current "
                "vertical direction."
            )
        direction = incoming_direction

        segment = _segment_between(
            ma, current_depth, interaction.depth, current_phase
        )
        if segment is None:
            raise ValueError(
                f"Cannot {interaction.kind} at the starting depth {interaction.depth} immediately."
            )
        ray_segments.append(segment)
        properties = _material_beyond_interaction(
            ma, interaction.depth, going_down=direction > 0.0
        )
        interaction_metadata.append(
            {
                "type": "reflection" if interaction.kind == "reflect" else "refraction",
                "depth": interaction.depth,
                "in_phase": current_phase,
                "out_phase": interaction.outgoing_phase,
                "seg_idx": len(ray_segments) - 1,
                "vp_beyond": properties["vp"],
                "vs_beyond": properties["vs"],
                "rho_beyond": properties["rho"],
            }
        )
        current_depth = interaction.depth
        current_phase = interaction.outgoing_phase
        if interaction.kind == "reflect":
            direction *= -1.0

    receiver_delta = receiver_depth - current_depth
    if abs(receiver_delta) > depth_tolerance:
        receiver_direction = 1.0 if receiver_delta > 0.0 else -1.0
        if direction is not None and receiver_direction != direction:
            raise ValueError(
                "Receiver depth is not reachable after the final itinerary interaction."
            )
        segment = _segment_between(ma, current_depth, receiver_depth, current_phase)
        if segment is not None:
            ray_segments.append(segment)

    return ray_segments, interaction_metadata


def _trace_one_many(
    ma: ModelArrays,
    src: np.ndarray,
    rcv: np.ndarray,
    source_phases: tuple[str, ...],
    refl_list: list[tuple[float, str]],
    refr_list: list[tuple[float, str]],
    ordered_interactions: tuple[Interaction, ...] | None,
    need_rays: bool,
    need_ray_parameters: bool,
    need_tstar: bool,
    need_spreading: bool,
    need_trans_product: bool,
    need_complex_coefficient_product: bool,
    need_diagnostics: bool,
    need_sensitivities: bool,
    transcoef_method: str,
    tol: float,
    max_iter: int,
) -> dict[str, tuple]:
    """Trace one source-receiver pair for multiple source phases."""
    sz = float(src[2])
    rz = float(rcv[2])
    can_share_direct = (
        not refl_list
        and not refr_list
        and not ordered_interactions
        and not need_complex_coefficient_product
    )
    cache: dict[str, tuple] = {}
    results = {}

    for source_phase in source_phases:
        _validate_sh_coupling(source_phase, refl_list + refr_list)
        family = velocity_key(source_phase)

        if can_share_direct and family in cache:
            tt, ray3d, p_val, tstar, spreading, _, complex_product, diagnostics, sensitivity = cache[family]
            trans = (
                _direct_trans_product(ma, sz, rz, source_phase, p_val, transcoef_method)
                if need_trans_product else None
            )
            results[source_phase] = (
                tt,
                ray3d,
                p_val if need_ray_parameters else None,
                tstar,
                spreading,
                trans,
                complex_product,
                diagnostics,
                sensitivity,
            )
            continue

        base = _trace_one(
            ma=ma,
            src=src,
            rcv=rcv,
            source_phase=source_phase,
            refl_list=refl_list,
            refr_list=refr_list,
            ordered_interactions=ordered_interactions,
            need_rays=need_rays,
            need_ray_parameters=(
                need_ray_parameters or need_trans_product or need_sensitivities
            ),
            need_tstar=need_tstar,
            need_spreading=need_spreading,
            need_trans_product=False if can_share_direct else need_trans_product,
            need_complex_coefficient_product=need_complex_coefficient_product,
            need_diagnostics=need_diagnostics,
            need_sensitivities=need_sensitivities,
            transcoef_method=transcoef_method,
            tol=tol,
            max_iter=max_iter,
        )

        if can_share_direct:
            cache[family] = base
            tt, ray3d, p_val, tstar, spreading, _, complex_product, diagnostics, sensitivity = base
            trans = (
                _direct_trans_product(ma, sz, rz, source_phase, p_val, transcoef_method)
                if need_trans_product else None
            )
            results[source_phase] = (
                tt,
                ray3d,
                p_val if need_ray_parameters else None,
                tstar,
                spreading,
                trans,
                complex_product,
                diagnostics,
                sensitivity,
            )
        else:
            results[source_phase] = base

    return results


def _trace_one(
    ma: ModelArrays,
    src: np.ndarray,
    rcv: np.ndarray,
    source_phase: str,
    refl_list: list[tuple[float, str]],
    refr_list: list[tuple[float, str]],
    ordered_interactions: tuple[Interaction, ...] | None,
    need_rays: bool,
    need_ray_parameters: bool,
    need_tstar: bool,
    need_spreading: bool,
    need_trans_product: bool,
    need_complex_coefficient_product: bool,
    need_diagnostics: bool,
    need_sensitivities: bool,
    transcoef_method: str,
    tol: float,
    max_iter: int,
) -> tuple:
    """Trace a single source->receiver ray."""
    source_phase = normalize_phase(source_phase)
    _validate_sh_coupling(source_phase, refl_list + refr_list)
    sx, sy, sz = float(src[0]), float(src[1]), float(src[2])
    rx, ry, rz = float(rcv[0]), float(rcv[1]), float(rcv[2])

    dx, dy = rx - sx, ry - sy
    epic = np.sqrt(dx * dx + dy * dy)

    if not refl_list and not refr_list and not ordered_interactions:
        stack = build_layer_stack(ma, sz, rz)
        vel = stack.v(velocity_key(source_phase))

        valid = stack.h > 1e-9
        if not np.any(valid):
            ray3d = np.array([[sx, sy, sz], [rx, ry, rz]]) if need_rays else None

            if epic < 1e-10:
                diagnostics = SolveDiagnostics(
                    converged=True,
                    method="q_newton",
                    iterations=0,
                    initial_q=0.0,
                    final_q=0.0,
                    initial_ray_parameter=0.0,
                    ray_parameter=0.0,
                    signed_offset_residual=0.0,
                    absolute_offset_residual=0.0,
                    conditioning=1.0,
                    criticality_margin=1.0,
                ) if need_diagnostics else None
                sensitivity = (
                    compute_ray_sensitivity(
                        ma, src, rcv, source_phase, [], 0.0
                    )
                    if need_sensitivities else None
                )
                return (
                    0.0,
                    ray3d,
                    0.0 if need_ray_parameters else None,
                    0.0 if need_tstar else None,
                    0.0 if need_spreading else None,
                    1.0 if need_trans_product else None,
                    1.0 + 0.0j if need_complex_coefficient_product else None,
                    diagnostics,
                    sensitivity,
                )

            v_hz = float(vel[0])
            tt_hz = epic / v_hz
            p_hz = 1.0 / v_hz

            q_arr = stack.qp if source_phase == "P" else stack.qs
            if need_tstar:
                q_name = "Qp" if source_phase == "P" else "Qs"
                if (
                    q_arr is None
                    or not np.isfinite(q_arr[0])
                    or q_arr[0] <= 0.0
                ):
                    raise ValueError(
                        f"{q_name} is required and must be positive when tstar "
                        "is requested."
                    )
                tstar_hz = float(epic / (v_hz * q_arr[0]))
            else:
                tstar_hz = None
            spreading_hz = epic * v_hz if need_spreading else None
            trans_hz = 1.0 if need_trans_product else None
            complex_hz = 1.0 + 0.0j if need_complex_coefficient_product else None
            diagnostics_hz = SolveDiagnostics(
                converged=True,
                method="q_newton",
                iterations=0,
                initial_q=np.inf,
                final_q=np.inf,
                initial_ray_parameter=p_hz,
                ray_parameter=p_hz,
                signed_offset_residual=0.0,
                absolute_offset_residual=0.0,
                conditioning=0.0,
                criticality_margin=0.0,
            ) if need_diagnostics else None
            sensitivity_hz = (
                compute_ray_sensitivity(
                    ma, src, rcv, source_phase, [], p_hz
                )
                if need_sensitivities else None
            )

            return (
                tt_hz,
                ray3d,
                p_hz if need_ray_parameters else None,
                tstar_hz,
                spreading_hz,
                trans_hz,
                complex_hz,
                diagnostics_hz,
                sensitivity_hz,
            )

        h_f = stack.h[valid]
        v_f = vel[valid]

        seg = {
            "h": h_f,
            "v": v_f,
            "vp": stack.vp[valid],
            "vs": stack.vs[valid],
            "rho": stack.rho[valid] if stack.rho is not None else None,
            "qp": stack.qp[valid] if stack.qp is not None else None,
            "qs": stack.qs[valid] if stack.qs is not None else None,
            "phase": source_phase,
            "start_z": sz,
            "end_z": rz,
        }

        res = solve(
            h=h_f,
            v=v_f,
            segments=[seg],
            interactions=[],
            epicentral_dist=epic,
            z_src=sz,
            z_rcv=rz,
            return_ray_path=need_rays,
            need_ray_parameter=need_ray_parameters or need_sensitivities,
            need_tstar=need_tstar,
            need_spreading=need_spreading,
            need_trans_product=need_trans_product,
            need_complex_coefficient_product=need_complex_coefficient_product,
            need_diagnostics=need_diagnostics,
            transcoef_method=transcoef_method,
            tol=tol,
            max_iter=max_iter,
        )

        ray3d = _project_ray_to_3d(res.ray_path, sx, sy, dx, dy, epic)
        sensitivity = (
            compute_ray_sensitivity(
                ma, src, rcv, source_phase, [seg], res.ray_parameter
            )
            if need_sensitivities else None
        )
        return (
            res.travel_time,
            ray3d,
            res.ray_parameter,
            res.tstar,
            res.spreading,
            res.trans_product,
            res.complex_coefficient_product,
            res.diagnostics,
            sensitivity,
        )

    if ordered_interactions:
        ray_segments, inter_meta = _compile_ordered_itinerary(
            ma, sz, rz, source_phase, ordered_interactions
        )
        if not ray_segments:
            raise ValueError("The itinerary does not contain a finite-length path.")

        all_h = np.concatenate([segment["h"] for segment in ray_segments])
        all_v = np.concatenate([segment["v"] for segment in ray_segments])
        res = solve(
            h=all_h,
            v=all_v,
            segments=ray_segments,
            interactions=inter_meta,
            epicentral_dist=epic,
            z_src=sz,
            z_rcv=rz,
            return_ray_path=need_rays,
            need_ray_parameter=need_ray_parameters or need_sensitivities,
            need_tstar=need_tstar,
            need_spreading=need_spreading,
            need_trans_product=need_trans_product,
            need_complex_coefficient_product=need_complex_coefficient_product,
            need_diagnostics=need_diagnostics,
            transcoef_method=transcoef_method,
            tol=tol,
            max_iter=max_iter,
        )
        ray3d = _project_ray_to_3d(res.ray_path, sx, sy, dx, dy, epic)
        sensitivity = (
            compute_ray_sensitivity(
                ma, src, rcv, source_phase, ray_segments, res.ray_parameter
            )
            if need_sensitivities else None
        )
        return (
            res.travel_time,
            ray3d,
            res.ray_parameter,
            res.tstar,
            res.spreading,
            res.trans_product,
            res.complex_coefficient_product,
            res.diagnostics,
            sensitivity,
        )

    ray_segments = []
    curr_z = sz
    curr_ph = source_phase
    directional_targets = [(z, ph) for z, ph in refl_list]
    itinerary_points = directional_targets + [(rz, None)]
    inter_meta = []

    for target_z, target_ph_after_turn in itinerary_points:
        going_down = target_z >= curr_z

        relevant_refr = []
        for r_z, r_ph in refr_list:
            if going_down:
                if curr_z < r_z < target_z:
                    relevant_refr.append((r_z, r_ph))
            else:
                if target_z < r_z < curr_z:
                    relevant_refr.append((r_z, r_ph))

        if going_down:
            relevant_refr.sort(key=lambda x: x[0])
        else:
            relevant_refr.sort(key=lambda x: x[0], reverse=True)

        sub_targets = relevant_refr + [(target_z, target_ph_after_turn)]

        for sub_z, sub_out_phase in sub_targets:
            stack = build_layer_stack(ma, curr_z, sub_z)
            vel = stack.v(velocity_key(curr_ph))
            valid_mask = stack.h > 1e-9

            if np.any(valid_mask):
                ray_segments.append({
                    "h": stack.h[valid_mask],
                    "v": vel[valid_mask],
                    "vp": stack.vp[valid_mask],
                    "vs": stack.vs[valid_mask],
                    "rho": stack.rho[valid_mask] if stack.rho is not None else None,
                    "qp": stack.qp[valid_mask] if stack.qp is not None else None,
                    "qs": stack.qs[valid_mask] if stack.qs is not None else None,
                    "phase": curr_ph,
                    "start_z": curr_z,
                    "end_z": sub_z,
                })

            is_major_turn = (sub_z == target_z) and (target_ph_after_turn is not None or target_z == rz)

            def _get_material_props(z_int: float, is_down_interaction: bool) -> dict[str, float]:
                delta = 1.0
                if is_down_interaction:
                    p_stack = build_layer_stack(ma, z_int, z_int + delta)
                else:
                    p_stack = build_layer_stack(ma, z_int - delta, z_int)
                return {
                    "vp": float(p_stack.vp[0]),
                    "vs": float(p_stack.vs[0]),
                    "rho": float(p_stack.rho[0]) if p_stack.rho is not None else 0.0,
                }

            if is_major_turn:
                if target_ph_after_turn is not None:
                    props_beyond = _get_material_props(sub_z, going_down)
                    seg_idx = len(ray_segments) - 1
                    if seg_idx < 0:
                        raise ValueError(f"Cannot reflect at the starting depth {sub_z} immediately.")

                    inter_meta.append({
                        "type": "reflection",
                        "depth": sub_z,
                        "in_phase": curr_ph,
                        "out_phase": target_ph_after_turn,
                        "seg_idx": seg_idx,
                        "vp_beyond": props_beyond["vp"],
                        "vs_beyond": props_beyond["vs"],
                        "rho_beyond": props_beyond["rho"],
                    })
                    curr_ph = target_ph_after_turn
            else:
                props_beyond = _get_material_props(sub_z, going_down)
                seg_idx = len(ray_segments) - 1
                if seg_idx < 0:
                    raise ValueError(f"Cannot refract at the starting depth {sub_z} immediately.")

                inter_meta.append({
                    "type": "refraction",
                    "depth": sub_z,
                    "in_phase": curr_ph,
                    "out_phase": sub_out_phase,
                    "seg_idx": seg_idx,
                    "vp_beyond": props_beyond["vp"],
                    "vs_beyond": props_beyond["vs"],
                    "rho_beyond": props_beyond["rho"],
                })
                curr_ph = sub_out_phase

            curr_z = sub_z

    if len(ray_segments) == 0:
        ray3d = np.array([[sx, sy, sz], [rx, ry, rz]]) if need_rays else None
        is_same_point = epic < 1e-10 and abs(sz - rz) < 1e-10
        return (
            0.0 if is_same_point else np.nan,
            ray3d,
            np.nan if need_ray_parameters else None,
            np.nan if need_tstar else None,
            np.nan if need_spreading else None,
            np.nan if need_trans_product else None,
            np.nan + 0.0j if need_complex_coefficient_product else None,
            None,
            compute_ray_sensitivity(ma, src, rcv, source_phase, [], np.nan)
            if need_sensitivities else None,
        )

    all_h = np.concatenate([s["h"] for s in ray_segments])
    all_v = np.concatenate([s["v"] for s in ray_segments])

    res = solve(
        h=all_h,
        v=all_v,
        segments=ray_segments,
        interactions=inter_meta,
        epicentral_dist=epic,
        z_src=sz,
        z_rcv=rz,
        return_ray_path=need_rays,
        need_ray_parameter=need_ray_parameters or need_sensitivities,
        need_tstar=need_tstar,
        need_spreading=need_spreading,
        need_trans_product=need_trans_product,
        need_complex_coefficient_product=need_complex_coefficient_product,
        need_diagnostics=need_diagnostics,
        transcoef_method=transcoef_method,
        tol=tol,
        max_iter=max_iter,
    )

    ray3d = _project_ray_to_3d(res.ray_path, sx, sy, dx, dy, epic)
    sensitivity = (
        compute_ray_sensitivity(
            ma, src, rcv, source_phase, ray_segments, res.ray_parameter
        )
        if need_sensitivities else None
    )
    return (
        res.travel_time,
        ray3d,
        res.ray_parameter,
        res.tstar,
        res.spreading,
        res.trans_product,
        res.complex_coefficient_product,
        res.diagnostics,
        sensitivity,
    )


def _unpack_results(
    results: list,
    requested: frozenset[str],
    source_phase: str | None = None,
) -> TraceResult:
    """Unpack a flat list of per-ray result tuples into a TraceResult."""

    def _maybe_array(values):
        if all(v is None for v in values):
            return None
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    def _value(result, index):
        return result[index] if len(result) > index else None

    tt = np.array([r[0] for r in results], dtype=float)
    rays = [r[1] for r in results] if "rays" in requested else None
    p_arr = _maybe_array([r[2] for r in results]) if "ray_parameters" in requested else None
    tstar = _maybe_array([r[3] for r in results]) if "tstar" in requested else None
    spreading = _maybe_array([r[4] for r in results]) if "spreading" in requested else None
    trans_product = _maybe_array([r[5] for r in results]) if "trans_product" in requested else None
    complex_product = (
        np.array(
            [np.nan + 0.0j if _value(r, 6) is None else _value(r, 6) for r in results],
            dtype=np.complex128,
        )
        if "complex_coefficient_product" in requested else None
    )
    diagnostics = (
        [_value(r, 7) for r in results] if "diagnostics" in requested else None
    )
    sensitivities = (
        [_value(r, 8) for r in results] if "sensitivities" in requested else None
    )

    return TraceResult(
        travel_times=tt,
        rays=rays,
        ray_parameters=p_arr,
        tstar=tstar,
        spreading=spreading,
        trans_product=trans_product,
        source_phase=source_phase,
        complex_coefficient_product=complex_product,
        diagnostics=diagnostics,
        sensitivities=sensitivities,
    )


def _unpack_multi_results(
    results: list[dict[str, tuple]],
    requested: frozenset[str],
    source_phases: tuple[str, ...],
) -> dict[str, TraceResult]:
    """Unpack per-ray multi-phase dictionaries into per-phase TraceResult objects."""
    return {
        phase: _unpack_results([ray_result[phase] for ray_result in results], requested, phase)
        for phase in source_phases
    }


def trace_rays(
    sources: np.ndarray,
    receivers: np.ndarray,
    velocity_df: pd.DataFrame,
    source_phase: str | Sequence[str] = "P",
    reflection: Sequence[tuple[float, str]] | None = None,
    refraction: Sequence[tuple[float, str]] | None = None,
    requested: Sequence[str] | None = DEFAULT_REQUESTED,
    transcoef_method: str = "standard",
    n_jobs: int = -1,
    backend: str = "loky",
    sequential_limit: int = 10_000,
    rays_per_chunk: int | None = None,
    tol: float = 1e-4,
    max_iter: int = 10,
    verbose: bool = True,
    itinerary: RayItinerary | None = None,
) -> TraceResult | dict[str, TraceResult]:
    r"""Trace rays for all source-receiver pairs.

    Every source is paired with every receiver, producing
    ``n_src x n_rcv`` rays (each source traced to all receivers).

    Parameters
    ----------
    sources : numpy.ndarray
        Source coordinates, shape ``(n_src, 3)`` or ``(3,)``.
    receivers : numpy.ndarray
        Receiver coordinates, shape ``(n_rcv, 3)`` or ``(3,)``.
    velocity_df : pandas.DataFrame
        Velocity model with columns ``Depth``, ``Vp``, ``Vs`` and
        optionally ``Rho``, ``Qp``, ``Qs``.
    source_phase : str or sequence of str
        Initial wave phase(s) at source: ``'P'``, ``'SV'``, ``'SH'``,
        or legacy ``'S'`` (alias for ``'SV'``). If a sequence is
        provided, a dictionary of ``TraceResult`` objects keyed by
        canonical phase is returned.
    reflection : list of (depth, phase), optional
        Reflection points as ``(depth, out_phase)`` tuples.
    refraction : list of (depth, phase), optional
        Refraction / mode-conversion points as ``(depth, out_phase)`` tuples.
    requested : sequence of str, optional
        Explicit set of requested outputs. Valid names are
        ``travel_times``, ``rays``, ``ray_parameters``, ``tstar``,
        ``spreading``, ``trans_product``, ``complex_coefficient_product``,
        ``diagnostics``, and ``sensitivities``. The set must include
        ``travel_times``. Diagnostic and sensitivity records are allocated
        only when explicitly requested.
    transcoef_method : str
        ``'standard'`` (Zoeppritz) or ``'normalized'``. The legacy
        spelling ``'angle'`` is an alias for ``'standard'``.
    n_jobs : int
        Number of parallel jobs (``-1`` = all physical cores).
    backend : str
        Joblib parallel backend (default ``'loky'``).
    sequential_limit : int
        If the total number of rays is below this threshold, run
        sequentially to avoid parallel overhead.
    rays_per_chunk : int or None
        Maximum number of rays to process per memory-bounded chunk.
    tol : float
        Newton convergence tolerance (m).
    max_iter : int
        Maximum Newton iterations.
    verbose : bool
        If *True*, print progress information for chunked processing.
    itinerary : RayItinerary, optional
        Explicit ordered interaction sequence. It cannot be combined with
        legacy ``reflection`` or ``refraction`` arguments. The itinerary's
        source phase is authoritative; a non-default explicit
        ``source_phase`` must agree with it.

    Returns
    -------
    TraceResult or dict of TraceResult
    """
    ordered_interactions: tuple[Interaction, ...] | None = None
    if itinerary is not None:
        if not isinstance(itinerary, RayItinerary):
            raise TypeError("itinerary must be a RayItinerary object.")
        if reflection is not None or refraction is not None:
            raise ValueError(
                "itinerary cannot be combined with reflection or refraction."
            )
        if not isinstance(source_phase, str):
            raise ValueError("itinerary cannot be combined with multiple source phases.")
        supplied_phase = normalize_phase(source_phase)
        if supplied_phase != "P" and supplied_phase != itinerary.source_phase:
            raise ValueError("source_phase must agree with itinerary.source_phase.")
        source_phase = itinerary.source_phase
        ordered_interactions = tuple(itinerary.interactions)

    source_phases, return_dict = _normalize_source_phases(source_phase)
    requested_set = _normalize_requested(requested)
    need_rays = "rays" in requested_set
    need_ray_parameters = "ray_parameters" in requested_set
    need_tstar = "tstar" in requested_set
    need_spreading = "spreading" in requested_set
    need_trans_product = "trans_product" in requested_set
    need_complex_coefficient_product = "complex_coefficient_product" in requested_set
    need_diagnostics = "diagnostics" in requested_set
    need_sensitivities = "sensitivities" in requested_set

    sources = np.atleast_2d(sources)
    receivers = np.atleast_2d(receivers)
    n_src = sources.shape[0]
    n_rcv = receivers.shape[0]
    n_rays = n_src * n_rcv

    refl_list = _normalize_interaction_phases(reflection)
    refr_list = _normalize_interaction_phases(refraction)

    model_depths = velocity_df["Depth"].values
    tol_depth = 1e-6

    def _validate_depths(interactions, name):
        for z, ph in interactions:
            if not np.any(np.abs(model_depths - z) < tol_depth):
                raise ValueError(
                    f"Invalid {name} depth {z}. Must match a model interface: {model_depths}"
                )
            if name == "reflection" and z < tol_depth:
                raise ValueError(
                    "Reflection at the surface (z=0.0) is not currently supported "
                    "for physical amplitude calculations. Please use a shallow "
                    "internal interface instead."
                )
            normalize_phase(ph)

    _validate_depths(refl_list, "reflection")
    _validate_depths(refr_list, "refraction")
    if ordered_interactions is not None:
        explicit_reflections = [
            (interaction.depth, interaction.outgoing_phase)
            for interaction in ordered_interactions
            if interaction.kind == "reflect"
        ]
        explicit_transmissions = [
            (interaction.depth, interaction.outgoing_phase)
            for interaction in ordered_interactions
            if interaction.kind == "transmit"
        ]
        _validate_depths(explicit_reflections, "reflection")
        _validate_depths(explicit_transmissions, "transmission")
    for phase in source_phases:
        explicit_phases = (
            [(item.depth, item.outgoing_phase) for item in ordered_interactions]
            if ordered_interactions is not None else []
        )
        _validate_sh_coupling(phase, refl_list + refr_list + explicit_phases)

    refl_z = {z for z, _ in refl_list}
    refr_z = {z for z, _ in refr_list}
    common = refl_z.intersection(refr_z)
    if common:
        raise ValueError(f"Cannot strictly reflect and refract at the same depth(s): {common}")

    ma = ModelArrays.from_dataframe(velocity_df)

    common_kw = dict(
        ma=ma,
        source_phases=source_phases,
        return_dict=return_dict,
        refl_list=refl_list,
        refr_list=refr_list,
        ordered_interactions=ordered_interactions,
        need_rays=need_rays,
        need_ray_parameters=need_ray_parameters,
        need_tstar=need_tstar,
        need_spreading=need_spreading,
        need_trans_product=need_trans_product,
        need_complex_coefficient_product=need_complex_coefficient_product,
        need_diagnostics=need_diagnostics,
        need_sensitivities=need_sensitivities,
        transcoef_method=transcoef_method,
        tol=tol,
        max_iter=max_iter,
    )

    if n_rays <= sequential_limit or n_jobs == 1:
        if return_dict:
            results = [
                _trace_one_many(
                    src=sources[i],
                    rcv=receivers[j],
                    ma=ma,
                    source_phases=source_phases,
                    refl_list=refl_list,
                    refr_list=refr_list,
                    ordered_interactions=ordered_interactions,
                    need_rays=need_rays,
                    need_ray_parameters=need_ray_parameters,
                    need_tstar=need_tstar,
                    need_spreading=need_spreading,
                    need_trans_product=need_trans_product,
                    need_complex_coefficient_product=need_complex_coefficient_product,
                    need_diagnostics=need_diagnostics,
                    need_sensitivities=need_sensitivities,
                    transcoef_method=transcoef_method,
                    tol=tol,
                    max_iter=max_iter,
                )
                for i in range(n_src)
                for j in range(n_rcv)
            ]
            return _unpack_multi_results(results, requested_set, source_phases)

        results = [
            _trace_one(
                src=sources[i],
                rcv=receivers[j],
                ma=ma,
                source_phase=source_phases[0],
                refl_list=refl_list,
                refr_list=refr_list,
                ordered_interactions=ordered_interactions,
                need_rays=need_rays,
                need_ray_parameters=need_ray_parameters,
                need_tstar=need_tstar,
                need_spreading=need_spreading,
                need_trans_product=need_trans_product,
                need_complex_coefficient_product=need_complex_coefficient_product,
                need_diagnostics=need_diagnostics,
                need_sensitivities=need_sensitivities,
                transcoef_method=transcoef_method,
                tol=tol,
                max_iter=max_iter,
            )
            for i in range(n_src)
            for j in range(n_rcv)
        ]
        return _unpack_results(results, requested_set, source_phases[0])

    if n_jobs == -1:
        n_workers = min(psutil.cpu_count(logical=False) or 4, n_rays)
    elif n_jobs < 0:
        n_workers = max(1, (psutil.cpu_count(logical=False) or 4) + n_jobs + 1)
    else:
        n_workers = n_jobs

    if rays_per_chunk is None:
        available_mem = psutil.virtual_memory().available
        bytes_per_ray = 64
        if need_rays:
            bytes_per_ray += 200
        if need_ray_parameters:
            bytes_per_ray += 8
        if need_tstar:
            bytes_per_ray += 8
        if need_spreading:
            bytes_per_ray += 8
        if need_trans_product:
            bytes_per_ray += 8
        if need_complex_coefficient_product:
            bytes_per_ray += 16
        if need_diagnostics:
            bytes_per_ray += 96
        if need_sensitivities:
            bytes_per_ray += 256
        usable_mem = available_mem * 0.5 / n_workers
        rays_per_chunk = max(100_000, int(usable_mem / bytes_per_ray))
        if verbose:
            print(
                f"Auto-detected rays_per_chunk: {rays_per_chunk:,} "
                f"(based on {available_mem / 1e9:.1f} GB available RAM)"
            )

    def _make_batches(index_pairs, src_arr, rcv_arr):
        batch_size = max(1, len(index_pairs) // n_workers)
        batches = []
        for i in range(0, len(index_pairs), batch_size):
            chunk = index_pairs[i:i + batch_size]
            batches.append((
                chunk,
                src_arr,
                rcv_arr,
                ma,
                source_phases,
                return_dict,
                refl_list,
                refr_list,
                ordered_interactions,
                need_rays,
                need_ray_parameters,
                need_tstar,
                need_spreading,
                need_trans_product,
                need_complex_coefficient_product,
                need_diagnostics,
                need_sensitivities,
                transcoef_method,
                tol,
                max_iter,
            ))
        return batches

    if n_rays > rays_per_chunk:
        rcv_per_chunk = max(1, rays_per_chunk // n_src)
        n_chunks = (n_rcv + rcv_per_chunk - 1) // rcv_per_chunk

        if verbose:
            print(
                f"Total rays: {n_rays:,} - processing in {n_chunks} chunks "
                f"({rcv_per_chunk:,} receivers per chunk)..."
            )

        def _empty_arrays():
            return {
                "tt": np.empty(n_rays, dtype=np.float64),
                "rays": [None] * n_rays if need_rays else None,
                "p": np.full(n_rays, np.nan, dtype=np.float64) if need_ray_parameters else None,
                "tstar": np.full(n_rays, np.nan, dtype=np.float64) if need_tstar else None,
                "spread": np.full(n_rays, np.nan, dtype=np.float64) if need_spreading else None,
                "trans": np.full(n_rays, np.nan, dtype=np.float64) if need_trans_product else None,
                "complex": np.full(n_rays, np.nan + 0.0j, dtype=np.complex128)
                if need_complex_coefficient_product else None,
                "diagnostics": [None] * n_rays if need_diagnostics else None,
                "sensitivities": [None] * n_rays if need_sensitivities else None,
            }

        phase_arrays = (
            {phase: _empty_arrays() for phase in source_phases}
            if return_dict else None
        )
        arrays = _empty_arrays() if not return_dict else None

        chunk_times: list[float] = []
        total_start = time.time()

        for chunk_idx in range(n_chunks):
            chunk_start = time.time()

            rcv_start = chunk_idx * rcv_per_chunk
            rcv_end = min((chunk_idx + 1) * rcv_per_chunk, n_rcv)
            chunk_rcv = receivers[rcv_start:rcv_end]
            chunk_nrcv = rcv_end - rcv_start

            chunk_pairs = [(i, j) for i in range(n_src) for j in range(chunk_nrcv)]
            batches = _make_batches(chunk_pairs, sources, chunk_rcv)

            batch_results = Parallel(
                n_jobs=n_workers, backend=backend, pre_dispatch="all"
            )(delayed(_trace_batch)(b) for b in batches)

            flat_idx = 0
            for batch_result in batch_results:
                for res in batch_result:
                    local_isrc = flat_idx // chunk_nrcv
                    local_ircv = flat_idx % chunk_nrcv
                    global_ircv = rcv_start + local_ircv
                    global_idx = local_isrc * n_rcv + global_ircv

                    if return_dict:
                        for phase in source_phases:
                            phase_res = res[phase]
                            arr = phase_arrays[phase]
                            arr["tt"][global_idx] = phase_res[0]
                            if arr["rays"] is not None:
                                arr["rays"][global_idx] = phase_res[1]
                            if arr["p"] is not None and phase_res[2] is not None:
                                arr["p"][global_idx] = phase_res[2]
                            if arr["tstar"] is not None and phase_res[3] is not None:
                                arr["tstar"][global_idx] = phase_res[3]
                            if arr["spread"] is not None and phase_res[4] is not None:
                                arr["spread"][global_idx] = phase_res[4]
                            if arr["trans"] is not None and phase_res[5] is not None:
                                arr["trans"][global_idx] = phase_res[5]
                            if arr["complex"] is not None and phase_res[6] is not None:
                                arr["complex"][global_idx] = phase_res[6]
                            if arr["diagnostics"] is not None:
                                arr["diagnostics"][global_idx] = phase_res[7]
                            if arr["sensitivities"] is not None:
                                arr["sensitivities"][global_idx] = phase_res[8]
                    else:
                        arrays["tt"][global_idx] = res[0]
                        if arrays["rays"] is not None:
                            arrays["rays"][global_idx] = res[1]
                        if arrays["p"] is not None and res[2] is not None:
                            arrays["p"][global_idx] = res[2]
                        if arrays["tstar"] is not None and res[3] is not None:
                            arrays["tstar"][global_idx] = res[3]
                        if arrays["spread"] is not None and res[4] is not None:
                            arrays["spread"][global_idx] = res[4]
                        if arrays["trans"] is not None and res[5] is not None:
                            arrays["trans"][global_idx] = res[5]
                        if arrays["complex"] is not None and res[6] is not None:
                            arrays["complex"][global_idx] = res[6]
                        if arrays["diagnostics"] is not None:
                            arrays["diagnostics"][global_idx] = res[7]
                        if arrays["sensitivities"] is not None:
                            arrays["sensitivities"][global_idx] = res[8]
                    flat_idx += 1

            chunk_elapsed = time.time() - chunk_start
            chunk_times.append(chunk_elapsed)
            if verbose:
                avg_t = sum(chunk_times) / len(chunk_times)
                remaining = avg_t * (n_chunks - chunk_idx - 1)
                if remaining >= 3600:
                    eta = f"{remaining / 3600:.1f}h"
                elif remaining >= 60:
                    eta = f"{remaining / 60:.1f}m"
                else:
                    eta = f"{remaining:.0f}s"
                print(
                    f"  Chunk {chunk_idx + 1}/{n_chunks} done "
                    f"({chunk_elapsed:.1f}s) - ETA: {eta}"
                )

            del chunk_pairs, batches, batch_results

        if verbose:
            total = time.time() - total_start
            if total >= 3600:
                ts = f"{total / 3600:.1f}h"
            elif total >= 60:
                ts = f"{total / 60:.1f}m"
            else:
                ts = f"{total:.1f}s"
            print(f"All chunks complete. Total time: {ts}")

        if return_dict:
            return {
                phase: TraceResult(
                    travel_times=arr["tt"],
                    rays=arr["rays"],
                    ray_parameters=arr["p"],
                    tstar=arr["tstar"],
                    spreading=arr["spread"],
                    trans_product=arr["trans"],
                    source_phase=phase,
                    complex_coefficient_product=arr["complex"],
                    diagnostics=arr["diagnostics"],
                    sensitivities=arr["sensitivities"],
                )
                for phase, arr in phase_arrays.items()
            }

        return TraceResult(
            travel_times=arrays["tt"],
            rays=arrays["rays"],
            ray_parameters=arrays["p"],
            tstar=arrays["tstar"],
            spreading=arrays["spread"],
            trans_product=arrays["trans"],
            source_phase=source_phases[0],
            complex_coefficient_product=arrays["complex"],
            diagnostics=arrays["diagnostics"],
            sensitivities=arrays["sensitivities"],
        )

    all_pairs = [(i, j) for i in range(n_src) for j in range(n_rcv)]
    batches = _make_batches(all_pairs, sources, receivers)

    batch_results = Parallel(
        n_jobs=n_workers, backend=backend, pre_dispatch="all"
    )(delayed(_trace_batch)(b) for b in batches)

    results: list = []
    for br in batch_results:
        results.extend(br)

    if return_dict:
        return _unpack_multi_results(results, requested_set, source_phases)

    return _unpack_results(results, requested_set, source_phases[0])
