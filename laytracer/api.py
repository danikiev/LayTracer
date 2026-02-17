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
        Geometrical spreading factor for each ray, shape ``(n_rays,)``.
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
    vel_type: str,
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

    # Build layer stack
    stack = build_layer_stack(vel_df, sz, rz)

    # Solve 2-D ray
    res = solve(
        stack,
        epicentral_dist=epic,
        z_src=sz,
        z_rcv=rz,
        vel_type=vel_type,
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
    vel_type: str = "Vp",
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
    vel_type : str
        ``'Vp'`` or ``'Vs'``.
    compute_amplitude : bool
        If *True*, compute :math:`t^*`, geometrical spreading and
        transmission coefficients alongside travel times.
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

    # Build argument list: each source paired with every receiver
    pairs = [
        (sources[i], receivers[j])
        for i in range(n_src)
        for j in range(n_rcv)
    ]

    common_kw = dict(
        vel_df=velocity_df,
        vel_type=vel_type,
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
