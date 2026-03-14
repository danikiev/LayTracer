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

from .model import ModelArrays, build_layer_stack
from .solver import solve


TRACE_OUTPUTS = frozenset({
    "travel_times",
    "rays",
    "ray_parameters",
    "tstar",
    "spreading",
    "trans_product",
})
DEFAULT_REQUESTED = ("travel_times", "rays", "ray_parameters")


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
        Product of transmission coefficients along each ray.
    """

    travel_times: np.ndarray
    rays: list[np.ndarray] | None = None
    ray_parameters: np.ndarray | None = None
    tstar: np.ndarray | None = None
    spreading: np.ndarray | None = None
    trans_product: np.ndarray | None = None


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


def _trace_batch(batch):
    """Worker function for parallel ray computation."""
    (
        batch_indices,
        source_coords,
        receiver_coords,
        model_arrays,
        source_phase,
        refl_list,
        refr_list,
        need_rays,
        need_ray_parameters,
        need_tstar,
        need_spreading,
        need_trans_product,
        transcoef_method,
        tol,
        max_iter,
    ) = batch

    results = []
    for isrc, ircv in batch_indices:
        results.append(
            _trace_one(
                ma=model_arrays,
                src=source_coords[isrc],
                rcv=receiver_coords[ircv],
                source_phase=source_phase,
                refl_list=refl_list,
                refr_list=refr_list,
                need_rays=need_rays,
                need_ray_parameters=need_ray_parameters,
                need_tstar=need_tstar,
                need_spreading=need_spreading,
                need_trans_product=need_trans_product,
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


def _trace_one(
    ma: ModelArrays,
    src: np.ndarray,
    rcv: np.ndarray,
    source_phase: str,
    refl_list: list[tuple[float, str]],
    refr_list: list[tuple[float, str]],
    need_rays: bool,
    need_ray_parameters: bool,
    need_tstar: bool,
    need_spreading: bool,
    need_trans_product: bool,
    transcoef_method: str,
    tol: float,
    max_iter: int,
) -> tuple[float, np.ndarray | None, float | None, float | None, float | None, float | None]:
    """Trace a single source->receiver ray."""
    sx, sy, sz = float(src[0]), float(src[1]), float(src[2])
    rx, ry, rz = float(rcv[0]), float(rcv[1]), float(rcv[2])

    dx, dy = rx - sx, ry - sy
    epic = np.sqrt(dx * dx + dy * dy)

    if not refl_list and not refr_list:
        stack = build_layer_stack(ma, sz, rz)
        vel = stack.v("Vp" if source_phase == "P" else "Vs")

        valid = stack.h > 1e-9
        if not np.any(valid):
            ray3d = np.array([[sx, sy, sz], [rx, ry, rz]]) if need_rays else None

            if epic < 1e-10:
                return (
                    0.0,
                    ray3d,
                    0.0 if need_ray_parameters else None,
                    0.0 if need_tstar else None,
                    0.0 if need_spreading else None,
                    1.0 if need_trans_product else None,
                )

            v_hz = float(vel[0])
            tt_hz = epic / v_hz
            p_hz = 1.0 / v_hz

            q_arr = stack.qp if source_phase == "P" else stack.qs
            tstar_hz = float(epic / (v_hz * q_arr[0])) if (need_tstar and q_arr is not None) else (0.0 if need_tstar else None)
            spreading_hz = epic * v_hz if need_spreading else None
            trans_hz = 1.0 if need_trans_product else None

            return (
                tt_hz,
                ray3d,
                p_hz if need_ray_parameters else None,
                tstar_hz,
                spreading_hz,
                trans_hz,
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
            need_ray_parameter=need_ray_parameters,
            need_tstar=need_tstar,
            need_spreading=need_spreading,
            need_trans_product=need_trans_product,
            transcoef_method=transcoef_method,
            tol=tol,
            max_iter=max_iter,
        )

        ray3d = _project_ray_to_3d(res.ray_path, sx, sy, dx, dy, epic)
        return (
            res.travel_time,
            ray3d,
            res.ray_parameter,
            res.tstar,
            res.spreading,
            res.trans_product,
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
            vel = stack.v("Vp" if curr_ph == "P" else "Vs")
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
        need_ray_parameter=need_ray_parameters,
        need_tstar=need_tstar,
        need_spreading=need_spreading,
        need_trans_product=need_trans_product,
        transcoef_method=transcoef_method,
        tol=tol,
        max_iter=max_iter,
    )

    ray3d = _project_ray_to_3d(res.ray_path, sx, sy, dx, dy, epic)
    return (
        res.travel_time,
        ray3d,
        res.ray_parameter,
        res.tstar,
        res.spreading,
        res.trans_product,
    )


def _unpack_results(results: list, requested: frozenset[str]) -> TraceResult:
    """Unpack a flat list of per-ray result tuples into a TraceResult."""

    def _maybe_array(values):
        if all(v is None for v in values):
            return None
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    tt = np.array([r[0] for r in results], dtype=float)
    rays = [r[1] for r in results] if "rays" in requested else None
    p_arr = _maybe_array([r[2] for r in results]) if "ray_parameters" in requested else None
    tstar = _maybe_array([r[3] for r in results]) if "tstar" in requested else None
    spreading = _maybe_array([r[4] for r in results]) if "spreading" in requested else None
    trans_product = _maybe_array([r[5] for r in results]) if "trans_product" in requested else None

    return TraceResult(
        travel_times=tt,
        rays=rays,
        ray_parameters=p_arr,
        tstar=tstar,
        spreading=spreading,
        trans_product=trans_product,
    )


def trace_rays(
    sources: np.ndarray,
    receivers: np.ndarray,
    velocity_df: pd.DataFrame,
    source_phase: str = "P",
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
) -> TraceResult:
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
    source_phase : str
        Initial wave phase at source: ``'P'`` or ``'S'``.
    reflection : list of (depth, phase), optional
        Reflection points as ``(depth, out_phase)`` tuples.
    refraction : list of (depth, phase), optional
        Refraction / mode-conversion points as ``(depth, out_phase)`` tuples.
    requested : sequence of str, optional
        Explicit set of requested outputs. Valid names are
        ``travel_times``, ``rays``, ``ray_parameters``, ``tstar``,
        ``spreading``, and ``trans_product``. The set must include
        ``travel_times``.
    transcoef_method : str
        ``'standard'`` (Zoeppritz) or ``'normalized'``.
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

    Returns
    -------
    TraceResult
    """
    requested_set = _normalize_requested(requested)
    need_rays = "rays" in requested_set
    need_ray_parameters = "ray_parameters" in requested_set
    need_tstar = "tstar" in requested_set
    need_spreading = "spreading" in requested_set
    need_trans_product = "trans_product" in requested_set

    sources = np.atleast_2d(sources)
    receivers = np.atleast_2d(receivers)
    n_src = sources.shape[0]
    n_rcv = receivers.shape[0]
    n_rays = n_src * n_rcv

    def _norm_interaction(arg: Sequence[tuple[float, str]] | None) -> list[tuple[float, str]]:
        return [] if arg is None else list(arg)

    refl_list = _norm_interaction(reflection)
    refr_list = _norm_interaction(refraction)

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
            if ph.upper() not in ("P", "S"):
                raise ValueError(f"Invalid phase '{ph}' in {name}. Must be 'P' or 'S'.")

    _validate_depths(refl_list, "reflection")
    _validate_depths(refr_list, "refraction")

    refl_z = {z for z, _ in refl_list}
    refr_z = {z for z, _ in refr_list}
    common = refl_z.intersection(refr_z)
    if common:
        raise ValueError(f"Cannot strictly reflect and refract at the same depth(s): {common}")

    ma = ModelArrays.from_dataframe(velocity_df)

    common_kw = dict(
        ma=ma,
        source_phase=source_phase,
        refl_list=refl_list,
        refr_list=refr_list,
        need_rays=need_rays,
        need_ray_parameters=need_ray_parameters,
        need_tstar=need_tstar,
        need_spreading=need_spreading,
        need_trans_product=need_trans_product,
        transcoef_method=transcoef_method,
        tol=tol,
        max_iter=max_iter,
    )

    if n_rays <= sequential_limit or n_jobs == 1:
        results = [
            _trace_one(src=sources[i], rcv=receivers[j], **common_kw)
            for i in range(n_src)
            for j in range(n_rcv)
        ]
        return _unpack_results(results, requested_set)

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
                source_phase,
                refl_list,
                refr_list,
                need_rays,
                need_ray_parameters,
                need_tstar,
                need_spreading,
                need_trans_product,
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

        tt_all = np.empty(n_rays, dtype=np.float64)
        rays_all: list[np.ndarray | None] | None = [None] * n_rays if need_rays else None
        p_all = np.full(n_rays, np.nan, dtype=np.float64) if need_ray_parameters else None
        tstar_all = np.full(n_rays, np.nan, dtype=np.float64) if need_tstar else None
        spread_all = np.full(n_rays, np.nan, dtype=np.float64) if need_spreading else None
        trans_all = np.full(n_rays, np.nan, dtype=np.float64) if need_trans_product else None

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

                    tt_all[global_idx] = res[0]
                    if rays_all is not None:
                        rays_all[global_idx] = res[1]
                    if p_all is not None and res[2] is not None:
                        p_all[global_idx] = res[2]
                    if tstar_all is not None and res[3] is not None:
                        tstar_all[global_idx] = res[3]
                    if spread_all is not None and res[4] is not None:
                        spread_all[global_idx] = res[4]
                    if trans_all is not None and res[5] is not None:
                        trans_all[global_idx] = res[5]
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

        return TraceResult(
            travel_times=tt_all,
            rays=rays_all,
            ray_parameters=p_all,
            tstar=tstar_all,
            spreading=spread_all,
            trans_product=trans_all,
        )

    all_pairs = [(i, j) for i in range(n_src) for j in range(n_rcv)]
    batches = _make_batches(all_pairs, sources, receivers)

    batch_results = Parallel(
        n_jobs=n_workers, backend=backend, pre_dispatch="all"
    )(delayed(_trace_batch)(b) for b in batches)

    results: list = []
    for br in batch_results:
        results.extend(br)

    return _unpack_results(results, requested_set)
