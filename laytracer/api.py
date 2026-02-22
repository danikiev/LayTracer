r"""
High-level multi-ray tracing interface.

Provides :func:`trace_rays`, the main entry point for tracing all
source–receiver pairs through a 1-D layered velocity model, with
optional parallel execution using the ``loky`` backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed

from .model import LayerStack, build_layer_stack
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
#  Worker function (one source–receiver pair)
# ═══════════════════════════════════════════════════════════════════════

def _trace_one(
    vel_df: pd.DataFrame,
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

    # 1. Sort reflections by proximity/logic?
    # For now, we assume the user provides them in temporal order if there are multiple.
    # But for a simple 1-reflection case, order doesn't matter much unless we zig-zag.
    # We will trust the user's order in refl_list for the sequence of bounces.

    # 2. Construct the sequence of "targets" (layer interfaces to hit)
    # Start: (sz, source_phase) -> Refl1 -> Refl2 -> ... -> (rz, final_phase?)
    #
    # We need to build a "folded" stack.
    # The Ray path is a sequence of Segments.
    # Each Segment is a monotonic path from z_start to z_end with a constant phase (P or S).

    current_z = sz
    current_phase = source_phase
    
    # We will accumulate arrays for the solver
    h_total = []
    v_total = []
    lmd_total = []  # will compute after vmax
    
    # For amplitude: interactions at interfaces
    # We need to track: (depth, type, in_phase, out_phase, layer_abover, layer_below)
    interactions = [] 
    
    # Combine reflections and final receiver into a target list
    # logic: The ray goes from current -> refl1 -> refl2 -> receiver
    targets = []
    for z_refl, ph_next in refl_list:
        targets.append((z_refl, "reflect", ph_next))
    targets.append((rz, "arrival", None))

    # Helper to extract a stack for one leg
    full_stack = build_layer_stack(vel_df, -1e9, 1e9) # Get full model for efficient queries
    # actually build_layer_stack logic is handy, let's just use it per leg
    
    for i, (z_target, interaction_type, next_phase) in enumerate(targets):
        # Build stack for this leg
        leg_stack = build_layer_stack(vel_df, current_z, z_target)
        
        # Get velocity for current phase
        # Note: leg_stack.h is always top-to-bottom order for the layers covered.
        # But if we go UP, we traverse them bottom-to-top.
        # solver.offset() sums terms, so order of h/v in summation doesn't matter for X(q).
        # It DOES matter for ray tracing coordinates later.
        
        # We append simple arrays
        v_leg = leg_stack.v("Vp" if current_phase == "P" else "Vs")
        h_total.append(leg_stack.h)
        v_total.append(v_leg)
        
        # Handle the interaction at the END of this leg (if not arrival)
        if interaction_type == "reflect":
            # Record explicit reflection
            interactions.append({
                "type": "reflection",
                "depth": z_target,
                "in_phase": current_phase,
                "out_phase": next_phase,
                "cum_idx": sum(len(x) for x in h_total) # Index boundary in flattened array
            })
            current_phase = next_phase
        
        # For refraction/transmission within the leg:
        # We need to check if any interface in `refr_list` was crossed.
        # This is tricky because `build_layer_stack` abstracts the layers.
        # But `refr_list` implies a MODE CONVERSION at a specific depth during transmission.
        # If the ray passes through z_refr, we split the leg there?
        # NO, the user spec says: "refraction... determines further propagation".
        # So we should have treated refractions as targets too!
        
        # Let's rethink targets.
        # Both "reflection" and "refraction" arguments define control points where
        # phase MIGHT change and direction MIGHT change (reflection).
        # We should merge them into a sorted list of events?
        # No, reflection dictates direction reversal. Refraction dictates phase change.
        pass # Logic continues below...

    # RE-IMPLEMENTATION of loop with Refractions included
    # We must treat refractions as waypoints that split the path.
    # Refractions don't change direction (monotonicity), but change phase.
    
    waypoints = [] # (depth, type, out_phase)
    
    # 1. Reflections are mandatory waypoints affecting direction
    for z, ph in refl_list:
        waypoints.append((z, "reflect", ph))
        
    # 2. Refractions: strictly speaking, if we pass through a refraction depth,
    # we must switch phase. But we don't know IF we pass it yet?
    # Actually we do. We go from Current -> Refl1. If RefractZ is between them, we hit it.
    # We need to inject refraction waypoints between start/reflections/end.
    
    # ... This is getting complex to sort. 
    # Let's start with the sequence of directional targets (Reflections + Receiver).
    directional_targets = [(z, ph) for z, ph in refl_list] 
    
    # We iterate legs. Inside each leg (Start -> Target), we check for refractions.
    
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
            stack = build_layer_stack(vel_df, curr_z, sub_z)
            
            # Phase velocity
            vel = stack.v("Vp" if curr_ph == "P" else "Vs")
            
            # Store segment data
            # If going UP, build_layer_stack returns layers shallow->deep.
            # But the ray traverses them deep->shallow.
            # consistent with solver logic (offset summation), we just flatten.
            
            # Filter zero-thickness layers to avoid Vmax pollution
            valid_mask = stack.h > 1e-9
            
            # If all are zero (e.g. start==end), keep one but handle velocity carefully?
            # Actually, if start==end, h=0, offset=0. It contributes nothing to Newton solver.
            # Solver can handle empty arrays? No.
            # But "Same-layer" check in solve() handles N=1. 
            # If a segment is truly 0 length, we should just skip appending it?
            # But we need it for continuity of metadata?
            # The solver only assumes monotonic segments.
            # If valid_mask is empty, we skip appending arrays but keep z book-keeping.
            
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
                # Segment is effectively zero thickness (e.g. bounce on interface).
                # We don't add physical layers to the solver.
                # But we must update curr_z below.
                # However, if we skip adding to ray_segments, 'seg_idx' for interactions
                # might point to non-existent segment?
                # Interaction logic assumes it happens at end of `ray_segments[-1]`.
                # If we skip, the interaction will be attached to previous segment.
                # This could be correct (reflection happens at end of previous leg).
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
                    p_stack = build_layer_stack(vel_df, z_int, z_int + delta)
                else:
                    # Look at [z-delta, z]
                    p_stack = build_layer_stack(vel_df, z_int - delta, z_int)
                
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
                # We ARE passing through.
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
    tol: float = 1e-4,
    max_iter: int = 10,
) -> TraceResult:
    r"""Trace rays for all source–receiver pairs.

    Every source is paired with every receiver, producing
    ``n_src × n_rcv`` rays (each source traced to all receivers).

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
        Number of parallel jobs (``-1`` = all cores).
    backend : str
        Joblib parallel backend (default ``'loky'``).
    sequential_limit : int
        If the total number of rays is below this threshold, run
        sequentially to avoid parallel overhead.
    tol : float
        Newton convergence tolerance (m).
    max_iter : int
        Maximum Newton iterations.

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

    # Build argument list: each source paired with every receiver
    pairs = [
        (sources[i], receivers[j])
        for i in range(n_src)
        for j in range(n_rcv)
    ]

    common_kw = dict(
        vel_df=velocity_df,
        source_phase=source_phase,
        refl_list=refl_list,
        refr_list=refr_list,
        compute_amplitude=compute_amplitude,
        transcoef_method=transcoef_method,
        tol=tol,
        max_iter=max_iter,
    )

    if n_rays <= sequential_limit or n_jobs == 1:
        results = [
            _trace_one(src=s, rcv=r, **common_kw)
            for s, r in pairs
        ]
    else:
        # Parallel with loky
        if n_jobs == -1:
            n_jobs = min(psutil.cpu_count(logical=False) or 4, n_rays)

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_trace_one)(src=s, rcv=r, **common_kw)
            for s, r in pairs
        )

    # ── Unpack results ──
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
