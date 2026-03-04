r"""
High-level multi-ray tracing interface.

Provides :func:`trace_rays`, the main entry point for tracing all
source–receiver pairs through a 1-D layered velocity model, with
optional parallel execution using the ``loky`` backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed

from .model import LayerStack, ModelArrays, build_layer_stack
from .solver import RayResult, solve


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


# ═══════════════════════════════════════════════════════════════════════
#  Worker functions
# ═══════════════════════════════════════════════════════════════════════

def _trace_batch(batch):
    """Worker function for parallel ray computation — processes a batch.

    Receives all shared data packed into *batch* so that the ``loky``
    backend serialises it only once per worker (not once per ray).
    Uses :class:`ModelArrays` (numpy) instead of a DataFrame for
    lightweight pickling and avoids repeated column extraction.
    """
    (batch_indices, source_coords, receiver_coords, model_arrays,
     source_phase, refl_list, refr_list,
     compute_amplitude, transcoef_method, tol, max_iter) = batch

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
                compute_amplitude=compute_amplitude,
                transcoef_method=transcoef_method,
                tol=tol,
                max_iter=max_iter,
            )
        )
    return results


def _trace_one(
    ma: ModelArrays,
    src: np.ndarray,
    rcv: np.ndarray,
    source_phase: str,
    refl_list: list[tuple[float, str]],
    refr_list: list[tuple[float, str]],
    compute_amplitude: bool,
    transcoef_method: str,
    tol: float,
    max_iter: int,
) -> tuple[float, np.ndarray, float, float | None, float | None, float | None]:
    """Trace a single source→receiver ray.  Returns a plain tuple for
    efficient serialisation in parallel workers."""
    sx, sy, sz = float(src[0]), float(src[1]), float(src[2])
    rx, ry, rz = float(rcv[0]), float(rcv[1]), float(rcv[2])

    # Epicentral distance (horizontal)
    dx, dy = rx - sx, ry - sy
    epic = np.sqrt(dx * dx + dy * dy)

    # ── Fast path for direct waves (no reflections / refractions) ──
    # Avoids itinerary-loop overhead: one build_layer_stack call,
    # no dict wrapping, straight into the solver.
    if not refl_list and not refr_list:
        stack = build_layer_stack(ma, sz, rz)
        vel = stack.v("Vp" if source_phase == "P" else "Vs")

        # Filter zero-thickness layers
        valid = stack.h > 1e-9
        if not np.any(valid):
            ray3d = np.array([[sx, sy, sz], [rx, ry, rz]])

            if epic < 1e-10:
                # Degenerate ray: source and receiver at same point
                return (
                    0.0, ray3d, 0.0,
                    0.0 if compute_amplitude else None,
                    None if compute_amplitude else None,
                    1.0 if compute_amplitude else None,
                )

            # Same-depth horizontal ray: straight line within one layer.
            # All layers had zero thickness, meaning z_src ≈ z_rcv.
            # The ray travels horizontally through the layer at that depth.
            v_hz = float(vel[0])
            tt_hz = epic / v_hz
            p_hz = 1.0 / v_hz  # horizontal (grazing) incidence

            if compute_amplitude:
                q_arr = stack.qp if source_phase == "P" else stack.qs
                tstar_hz = float(epic / (v_hz * q_arr[0])) if q_arr is not None else 0.0
                spreading_hz = epic * v_hz  # L = r·v for homogeneous medium
                trans_hz = 1.0              # no interface crossed
            else:
                tstar_hz = None
                spreading_hz = None
                trans_hz = None

            return (tt_hz, ray3d, p_hz, tstar_hz, spreading_hz, trans_hz)

        h_f = stack.h[valid]
        v_f = vel[valid]

        # Build a single segment (avoid dict — use list-of-one)
        seg = {
            "h": h_f, "v": v_f,
            "vp": stack.vp[valid], "vs": stack.vs[valid],
            "rho": stack.rho[valid] if stack.rho is not None else None,
            "qp": stack.qp[valid] if stack.qp is not None else None,
            "qs": stack.qs[valid] if stack.qs is not None else None,
            "phase": source_phase, "start_z": sz, "end_z": rz,
        }

        res = solve(
            h=h_f, v=v_f,
            segments=[seg], interactions=[],
            epicentral_dist=epic, z_src=sz, z_rcv=rz,
            compute_amplitude=compute_amplitude,
            transcoef_method=transcoef_method,
            tol=tol, max_iter=max_iter,
        )

        ray2d = res.ray_path
        M = ray2d.shape[0]
        ray3d = np.empty((M, 3))
        if epic > 1e-10:
            ux, uy = dx / epic, dy / epic
        else:
            ux, uy = 1.0, 0.0
        ray3d[:, 0] = sx + ray2d[:, 0] * ux
        ray3d[:, 1] = sy + ray2d[:, 0] * uy
        ray3d[:, 2] = ray2d[:, 1]

        return (
            res.travel_time, ray3d, res.ray_parameter,
            res.tstar, res.spreading, res.trans_product,
        )

    # ── General path: reflections and/or refractions ──
    directional_targets = [(z, ph) for z, ph in refl_list] 
        
    ray_segments = [] # store (h, v, phase, direction_sign)
    
    curr_z = sz
    curr_ph = source_phase
    
    # Full itinerary points: [Start] -> [Refl1] -> [Refl2] -> [Receiver]
    itinerary_points = directional_targets + [(rz, None)]
    
    inter_meta = [] # metadata for amplitude
    
    for target_z, target_ph_after_turn in itinerary_points:
        # Direction for this major leg
        going_down = (target_z >= curr_z)
        
        # Identify any refractions that occur on this path
        # Filter refr_list for depths strictly between curr_z and target_z
        # If going down: curr_z < z_refr < target_z
        # If going up: target_z < z_refr < curr_z
        
        relevant_refr = []
        for r_z, r_ph in refr_list:
            if going_down:
                if curr_z < r_z < target_z:
                    relevant_refr.append((r_z, r_ph))
            else:
                if target_z < r_z < curr_z:
                    relevant_refr.append((r_z, r_ph))
        
        # Sort refractions by proximity to curr_z
        if going_down:
            relevant_refr.sort(key=lambda x: x[0])
        else:
            relevant_refr.sort(key=lambda x: x[0], reverse=True)
            
        # Build sub-legs
        sub_targets = relevant_refr + [(target_z, target_ph_after_turn)]
        
        for sub_z, sub_out_phase in sub_targets:
            # Build stack for this sub-segment
            # sub_z is the end of this sub-segment
            
            # Note: build_layer_stack is robust for z1 > z2 (computes thickness correctly)
            stack = build_layer_stack(ma, curr_z, sub_z)
            
            # Phase velocity
            vel = stack.v("Vp" if curr_ph == "P" else "Vs")
                        
            # Filter zero-thickness layers to avoid Vmax pollution
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
                    "end_z": sub_z
                })
            else:                
                pass

            # Check if this sub-target is a Refraction or the Major Turn
            is_major_turn = (sub_z == target_z) and (target_ph_after_turn is not None or target_z == rz)
            
            # Helper to get properties of the layer BEYOND the interface
            # If going down, "beyond" is layer starting at sub_z.
            # If going up, "beyond" is layer ending at sub_z.
            # We use build_layer_stack on a small interval to probe it.
            
            def _get_material_props(z_int, is_down_interaction):
                # Probe a tiny segment beyond the interface
                delta = 1.0 # 1 meter check
                if is_down_interaction:
                    # Look at [z, z+delta]
                    p_stack = build_layer_stack(ma, z_int, z_int + delta)
                else:
                    # Look at [z-delta, z]
                    p_stack = build_layer_stack(ma, z_int - delta, z_int)
                
                # Check if we are at model boundary (bottom/top)
                # build_layer_stack handles this gracefully generally, 
                # but let's be safe.
                # Just return the first layer props of the stack
                return {
                    "vp": float(p_stack.vp[0]),
                    "vs": float(p_stack.vs[0]),
                    "rho": float(p_stack.rho[0]) if p_stack.rho is not None else 0.0
                }

            if is_major_turn:
                 # This is the endpoint of the major leg (Reflection or Receiver)
                 if target_ph_after_turn is not None:
                     # Reflection
                     # "Beyond" side is the layer we WOULD have entered if we didn't turn.
                     # If we are going down, beyond is Below.
                     # If we are going up, beyond is Above (reflection from underside!).
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
                         "rho_beyond": props_beyond["rho"]
                     })
                     curr_ph = target_ph_after_turn
            else:
                # Refraction/Transmission                
                # "Beyond" side is the layer we are ABOUT to enter (which will be first layer of next segment).
                # We can fetch it now.
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
                     "rho_beyond": props_beyond["rho"]
                })
                curr_ph = sub_out_phase
            
            curr_z = sub_z

    # Flatten arrays for solver
    if len(ray_segments) == 0:
        # Degenerate case: source and receiver share the same depth,
        # or the path lies entirely outside the model.  Return a
        # minimal result (zero travel time, straight-line ray, NaN
        # amplitude quantities) so the caller can proceed.
        ray3d = np.array([[sx, sy, sz], [rx, ry, rz]])
        return (
            0.0 if epic < 1e-10 and abs(sz - rz) < 1e-10 else np.nan,
            ray3d,
            np.nan,
            np.nan if compute_amplitude else None,
            np.nan if compute_amplitude else None,
            np.nan if compute_amplitude else None,
        )

    all_h = np.concatenate([s["h"] for s in ray_segments])
    all_v = np.concatenate([s["v"] for s in ray_segments])
    
    # Solve 2-D ray    
    res = solve(
        h=all_h,
        v=all_v,
        segments=ray_segments,
        interactions=inter_meta,
        epicentral_dist=epic,
        z_src=sz,
        z_rcv=rz,
        compute_amplitude=compute_amplitude,
        transcoef_method=transcoef_method,
        tol=tol,
        max_iter=max_iter,
    )

    # ── Convert 2-D ray path to 3-D ──
    ray2d = res.ray_path  # (M, 2) — [x_ray, z]
    M = ray2d.shape[0]
    ray3d = np.empty((M, 3))

    if epic > 1e-10:
        ux, uy = dx / epic, dy / epic
    else:
        ux, uy = 1.0, 0.0

    ray3d[:, 0] = sx + ray2d[:, 0] * ux
    ray3d[:, 1] = sy + ray2d[:, 0] * uy
    ray3d[:, 2] = ray2d[:, 1]

    return (
        res.travel_time,
        ray3d,
        res.ray_parameter,
        res.tstar,
        res.spreading,
        res.trans_product,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def _unpack_results(
    results: list,
    compute_amplitude: bool,
) -> TraceResult:
    """Unpack a flat list of per-ray result tuples into a :class:`TraceResult`."""
    tt = np.array([r[0] for r in results])
    rays = [r[1] for r in results]
    p_arr = np.array([r[2] for r in results])

    tstar = None
    spreading = None
    trans_product = None

    if compute_amplitude:
        ts = [r[3] for r in results]
        sp = [r[4] for r in results]
        tp = [r[5] for r in results]
        tstar = np.array(ts, dtype=float) if ts[0] is not None else None
        spreading = np.array(sp, dtype=float) if sp[0] is not None else None
        trans_product = np.array(tp, dtype=float) if tp[0] is not None else None

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
    compute_amplitude: bool = False,
    transcoef_method: str = "angle",
    n_jobs: int = -1,
    backend: str = "loky",
    sequential_limit: int = 10_000,
    rays_per_chunk: int | None = None,
    tol: float = 1e-4,
    max_iter: int = 10,
    verbose: bool = True,
) -> TraceResult:
    r"""Trace rays for all source–receiver pairs.

    Every source is paired with every receiver, producing
    ``n_src × n_rcv`` rays (each source traced to all receivers).

    Parallel execution uses a **batched** dispatch strategy: rays are
    grouped into large batches (one per worker core) so that the
    velocity model is serialised only once per batch rather than once
    per ray.  For very large problems the work is further split into
    **memory-bounded chunks** whose size is automatically determined
    from available RAM (or set explicitly via *rays_per_chunk*).

    Parameters
    ----------
    sources : numpy.ndarray
        Source coordinates, shape ``(n_src, 3)`` or ``(3,)``
        for a single source.  Columns: ``[X, Y, Z]``.
    receivers : numpy.ndarray
        Receiver coordinates, shape ``(n_rcv, 3)`` or ``(3,)``.
    velocity_df : pandas.DataFrame
        Velocity model with columns ``Depth``, ``Vp``, ``Vs`` and
        optionally ``Rho``, ``Qp``, ``Qs``.
    source_phase : str
        Initial wave phase at source: ``'P'`` or ``'S'``.
    reflection : list of (depth, phase), optional
        List of reflection points. Each element is a tuple
        ``(depth, out_phase)`` where ``depth`` matches a layer
        interface in ``velocity_df`` and ``out_phase`` is the
        phase of the reflected wave (``'P'`` or ``'S'``).
        Example: ``[(2000.0, 'S')]`` for P-to-S reflection at 2km.
    refraction : list of (depth, phase), optional
        List of specific refraction/mode-conversion points.
        Each element is a tuple ``(depth, out_phase)``.
        If a depth is not listed here, transmission assumes
        preservation of the incident phase.
    compute_amplitude : bool
        If *True*, computes the travel time alongside the ray path, the attenuation operator
        :math:`t^*`, relative geometrical spreading, and Zoeppritz transmission
        products.
    transcoef_method : str
        ``'normal'`` or ``'angle'`` (Zoeppritz).
    n_jobs : int
        Number of parallel jobs (``-1`` = all physical cores).
    backend : str
        Joblib parallel backend (default ``'loky'``).
    sequential_limit : int
        If the total number of rays is below this threshold, run
        sequentially to avoid parallel overhead.
    rays_per_chunk : int or None
        Maximum number of rays to process per memory-bounded chunk.
        Larger values use more memory but have less overhead.  If
        *None* (default), the chunk size is automatically determined
        from available system memory.
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
    sources = np.atleast_2d(sources)
    receivers = np.atleast_2d(receivers)
    n_src = sources.shape[0]
    n_rcv = receivers.shape[0]
    n_rays = n_src * n_rcv

    # Helper: Normalize phase list to list of tuples
    def _norm_interaction(
        arg: Sequence[tuple[float, str]] | None
    ) -> list[tuple[float, str]]:
        if arg is None:
            return []
        return list(arg)

    refl_list = _norm_interaction(reflection)
    refr_list = _norm_interaction(refraction)

    # Validate depths against model
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
                    f"Reflection at the surface (z=0.0) is not currently supported for "
                    "physical amplitude calculations. Please use a shallow internal interface instead."
                )
            if ph.upper() not in ("P", "S"):
                raise ValueError(f"Invalid phase '{ph}' in {name}. Must be 'P' or 'S'.")

    _validate_depths(refl_list, "reflection")
    _validate_depths(refr_list, "refraction")

    # Check for conflicts (same depth in both lists)
    refl_z = {z for z, _ in refl_list}
    refr_z = {z for z, _ in refr_list}
    common = refl_z.intersection(refr_z)
    if common:
        raise ValueError(
            f"Cannot strictly reflect and refract at the same depth(s): {common}"
        )

    # ── Pre-extract arrays from DataFrame once (lightweight pickle) ──
    ma = ModelArrays.from_dataframe(velocity_df)

    common_kw = dict(
        ma=ma,
        source_phase=source_phase,
        refl_list=refl_list,
        refr_list=refr_list,
        compute_amplitude=compute_amplitude,
        transcoef_method=transcoef_method,
        tol=tol,
        max_iter=max_iter,
    )

    # ── Sequential path ──
    if n_rays <= sequential_limit or n_jobs == 1:
        results = [
            _trace_one(src=sources[i], rcv=receivers[j], **common_kw)
            for i in range(n_src)
            for j in range(n_rcv)
        ]
        return _unpack_results(results, compute_amplitude)

    # ── Determine worker count ──
    if n_jobs == -1:
        n_workers = min(psutil.cpu_count(logical=False) or 4, n_rays)
    elif n_jobs < 0:
        n_workers = max(1, (psutil.cpu_count(logical=False) or 4) + n_jobs + 1)
    else:
        n_workers = n_jobs

    # ── Auto-determine rays_per_chunk from available memory ──
    if rays_per_chunk is None:
        available_mem = psutil.virtual_memory().available
        # Estimate memory per ray (indices + result tuple + ray path array)
        bytes_per_ray = 64  # base: index tuple + travel-time scalar
        bytes_per_ray += 200  # ray path (typical ~8 points × 3 coords × 8 bytes)
        if compute_amplitude:
            bytes_per_ray += 200  # tstar, spreading, trans_product extras
        # Use 50% of available memory, divided by worker count
        usable_mem = available_mem * 0.5 / n_workers
        rays_per_chunk = max(100_000, int(usable_mem / bytes_per_ray))
        if verbose:
            print(
                f"Auto-detected rays_per_chunk: {rays_per_chunk:,} "
                f"(based on {available_mem / 1e9:.1f} GB available RAM)"
            )

    # ── Helper: build batches for a set of index pairs ──
    def _make_batches(index_pairs, src_arr, rcv_arr):
        """Split *index_pairs* into ~n_workers equal batches.

        Each batch carries the shared source/receiver arrays and
        :class:`ModelArrays` (numpy-only, fast to pickle) rather
        than a DataFrame."""
        batch_size = max(1, len(index_pairs) // n_workers)
        batches = []
        for i in range(0, len(index_pairs), batch_size):
            chunk = index_pairs[i : i + batch_size]
            batches.append((
                chunk,
                src_arr,
                rcv_arr,
                ma,
                source_phase,
                refl_list,
                refr_list,
                compute_amplitude,
                transcoef_method,
                tol,
                max_iter,
            ))
        return batches

    # ── Chunked processing for very large problems ──
    if n_rays > rays_per_chunk:
        # Chunk along the receiver axis to keep memory bounded
        rcv_per_chunk = max(1, rays_per_chunk // n_src)
        n_chunks = (n_rcv + rcv_per_chunk - 1) // rcv_per_chunk

        if verbose:
            print(
                f"Total rays: {n_rays:,} — processing in {n_chunks} chunks "
                f"({rcv_per_chunk:,} receivers per chunk)..."
            )

        # Pre-allocate flat result arrays
        tt_all = np.empty(n_rays, dtype=np.float64)
        p_all = np.empty(n_rays, dtype=np.float64)
        rays_all: list[np.ndarray | None] = [None] * n_rays
        tstar_all = np.empty(n_rays, dtype=np.float64) if compute_amplitude else None
        spread_all = np.empty(n_rays, dtype=np.float64) if compute_amplitude else None
        trans_all = np.empty(n_rays, dtype=np.float64) if compute_amplitude else None

        chunk_times: list[float] = []
        total_start = time.time()

        for chunk_idx in range(n_chunks):
            chunk_start = time.time()

            rcv_start = chunk_idx * rcv_per_chunk
            rcv_end = min((chunk_idx + 1) * rcv_per_chunk, n_rcv)
            chunk_rcv = receivers[rcv_start:rcv_end]
            chunk_nrcv = rcv_end - rcv_start

            # Build index pairs for this chunk (indices into sources / chunk_rcv)
            chunk_pairs = [
                (i, j)
                for i in range(n_src)
                for j in range(chunk_nrcv)
            ]

            batches = _make_batches(chunk_pairs, sources, chunk_rcv)

            batch_results = Parallel(
                n_jobs=n_workers, backend=backend, pre_dispatch="all"
            )(delayed(_trace_batch)(b) for b in batches)

            # Flatten and store at correct global indices
            flat_idx = 0
            for batch_result in batch_results:
                for res in batch_result:
                    local_isrc = flat_idx // chunk_nrcv
                    local_ircv = flat_idx % chunk_nrcv
                    global_ircv = rcv_start + local_ircv
                    global_idx = local_isrc * n_rcv + global_ircv

                    tt_all[global_idx] = res[0]
                    rays_all[global_idx] = res[1]
                    p_all[global_idx] = res[2]
                    if compute_amplitude:
                        if tstar_all is not None and res[3] is not None:
                            tstar_all[global_idx] = res[3]
                        if spread_all is not None and res[4] is not None:
                            spread_all[global_idx] = res[4]
                        if trans_all is not None and res[5] is not None:
                            trans_all[global_idx] = res[5]
                    flat_idx += 1

            # Timing / progress
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
                    f"({chunk_elapsed:.1f}s) — ETA: {eta}"
                )

            # Free intermediate memory
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

    # ── Standard batched parallel path (fits in one chunk) ──
    all_pairs = [
        (i, j)
        for i in range(n_src)
        for j in range(n_rcv)
    ]

    batches = _make_batches(all_pairs, sources, receivers)

    batch_results = Parallel(
        n_jobs=n_workers, backend=backend, pre_dispatch="all"
    )(delayed(_trace_batch)(b) for b in batches)

    # Flatten batches into a single result list
    results: list = []
    for br in batch_results:
        results.extend(br)

    return _unpack_results(results, compute_amplitude)
