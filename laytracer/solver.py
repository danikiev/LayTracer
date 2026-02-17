r"""
Core two-point ray tracing solver using the dimensionless *q*-parameter.

Implements the method of :footcite:t:`FangChen2019`:

*   Dimensionless ray parameter :math:`q` for numerical stability
*   Vectorised offset equation :math:`X(q)` and its derivatives
*   Asymptotic initial estimate (Section 2.2 of the paper)
*   Second-order Newton iteration (Section 2.3 of the paper)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.optimize as opt

from .model import LayerStack


# ═══════════════════════════════════════════════════════════════════════
#  Offset equation and derivatives  (Fang & Chen 2019, §2.1)
# ═══════════════════════════════════════════════════════════════════════

def offset(q: float, h: np.ndarray, lmd: np.ndarray) -> float:
    r"""Total horizontal range :math:`X(q)`.

    .. math::
        X(q) = \sum_{k=1}^{N}
               \frac{q\,\lambda_k\,h_k}
                    {\sqrt{1 + (1 - \lambda_k^2)\,q^2}}

    Parameters
    ----------
    q : float
        Dimensionless ray parameter.
    h : numpy.ndarray
        Layer thicknesses (m), shape ``(N,)``.
    lmd : numpy.ndarray
        Velocity ratios :math:`\lambda_k = v_k / v_{\max}`, shape ``(N,)``.

    Returns
    -------
    float
    """
    s = 1.0 - lmd * lmd          # (1 − λ²)
    return float(np.sum(q * lmd * h / np.sqrt(1.0 + s * q * q)))


def offset_dq(q: float, h: np.ndarray, lmd: np.ndarray) -> float:
    r"""First derivative :math:`\mathrm{d}X/\mathrm{d}q`.

    .. math::
        \frac{\mathrm{d}X}{\mathrm{d}q}
        = \sum_{k=1}^{N}
          \frac{\lambda_k\,h_k}
               {\bigl[1 + (1 - \lambda_k^2)\,q^2\bigr]^{3/2}}
    """
    s = 1.0 - lmd * lmd
    return float(np.sum(lmd * h / (1.0 + s * q * q) ** 1.5))


def offset_dq2(q: float, h: np.ndarray, lmd: np.ndarray) -> float:
    r"""Second derivative :math:`\mathrm{d}^2X/\mathrm{d}q^2`.

    .. math::
        \frac{\mathrm{d}^2X}{\mathrm{d}q^2}
        = -3\,q\,\sum_{k=1}^{N}
          \frac{(1 - \lambda_k^2)\,\lambda_k\,h_k}
               {\bigl[1 + (1 - \lambda_k^2)\,q^2\bigr]^{5/2}}
    """
    s = 1.0 - lmd * lmd
    return float(-3.0 * q * np.sum(s * lmd * h / (1.0 + s * q * q) ** 2.5))


# ═══════════════════════════════════════════════════════════════════════
#  Parameter conversions
# ═══════════════════════════════════════════════════════════════════════

def q_from_p(p: float, vmax: float) -> float:
    r"""Convert horizontal slowness *p* to dimensionless parameter *q*.

    .. math:: q = \frac{p\,v_{\max}}{\sqrt{1 - p^2\,v_{\max}^2}}
    """
    pv = p * vmax
    return pv / np.sqrt(1.0 - pv * pv)


def p_from_q(q: float, vmax: float) -> float:
    r"""Convert dimensionless parameter *q* to horizontal slowness *p*.

    .. math:: p = \frac{q}{v_{\max}\,\sqrt{1 + q^2}}
    """
    return q / (vmax * np.sqrt(1.0 + q * q))


# ═══════════════════════════════════════════════════════════════════════
#  Initial estimate  (Fang & Chen 2019, §2.2)
# ═══════════════════════════════════════════════════════════════════════

def initial_q(
    X_target: float,
    h: np.ndarray,
    lmd: np.ndarray,
) -> float:
    r"""Asymptotic initial estimate :math:`q_0` for the Newton iteration.

    Two linear asymptotes of :math:`X(q)`:

    *   **Near-field** (:math:`q\to 0`):
        :math:`X \approx m_0\,q`, where :math:`m_0 = \sum \lambda_k h_k`.

    *   **Far-field** (:math:`q\to\infty`):
        :math:`X \approx m_\infty q + b_\infty`, where
        :math:`m_\infty = \sum_{k:\,\lambda_k=1} h_k` and
        :math:`b_\infty = \sum_{k:\,\lambda_k<1}
        \frac{\lambda_k h_k}{\sqrt{1-\lambda_k^2}}`.

    The initial estimate is chosen as:

    *   :math:`q_0 = X / m_0` when :math:`X \le X^*` (near-field),
    *   :math:`q_0 = (X - b_\infty) / m_\infty` otherwise.

    where :math:`X^* = m_0\,q^*` and :math:`q^* = b_\infty/(m_0-m_\infty)`.
    """
    m0 = float(np.sum(lmd * h))  # near-field slope

    is_max = np.isclose(lmd, 1.0, atol=1e-12)
    m_inf = float(np.sum(h[is_max])) if np.any(is_max) else 0.0

    not_max = ~is_max
    if np.any(not_max):
        b_inf = float(
            np.sum(lmd[not_max] * h[not_max] / np.sqrt(1.0 - lmd[not_max] ** 2))
        )
    else:
        b_inf = 0.0

    # Default: near-field
    if m0 < 1e-15:
        return 1.0
    q0 = X_target / m0

    # Switch to far-field if X_target is beyond the crossover
    if m_inf > 0 and m0 - m_inf > 1e-12:
        q_cross = b_inf / (m0 - m_inf)
        X_cross = m0 * q_cross
        if X_target > X_cross and m_inf > 1e-12:
            q0 = (X_target - b_inf) / m_inf

    return max(q0, 1e-10)


# ═══════════════════════════════════════════════════════════════════════
#  Second-order Newton iteration  (Fang & Chen 2019, §2.3)
# ═══════════════════════════════════════════════════════════════════════

def newton_step(
    q_i: float,
    X_target: float,
    h: np.ndarray,
    lmd: np.ndarray,
) -> tuple[float, float]:
    r"""One second-order (quadratic) Newton step.

    Solves the local quadratic approximation

    .. math::
        \tfrac{1}{2}\,X''(q_i)\,\Delta q^2
        + X'(q_i)\,\Delta q
        + \bigl[X(q_i) - X_R\bigr] = 0

    and selects the root that minimises :math:`|X(q_i+\Delta q)-X_R|`.

    Parameters
    ----------
    q_i : float
        Current iterate.
    X_target : float
        Desired horizontal range.
    h, lmd : numpy.ndarray
        Layer thicknesses and velocity ratios.

    Returns
    -------
    q_new : float
        Updated iterate.
    X_new : float
        Offset at ``q_new``.
    """
    X_i = offset(q_i, h, lmd)
    C = X_i - X_target
    B = offset_dq(q_i, h, lmd)
    A = 0.5 * offset_dq2(q_i, h, lmd)

    # Linear fallback
    if abs(A) < 1e-15:
        dq = -C / B if abs(B) > 1e-15 else 0.0
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0:
            dq = -C / B if abs(B) > 1e-15 else 0.0
        else:
            sd = np.sqrt(disc)
            dq1 = (-B + sd) / (2.0 * A)
            dq2 = (-B - sd) / (2.0 * A)

            q1, q2 = q_i + dq1, q_i + dq2
            ok1, ok2 = q1 > 0, q2 > 0
            if ok1 and ok2:
                e1 = abs(offset(q1, h, lmd) - X_target)
                e2 = abs(offset(q2, h, lmd) - X_target)
                dq = dq1 if e1 <= e2 else dq2
            elif ok1:
                dq = dq1
            elif ok2:
                dq = dq2
            else:
                dq = -C / B if abs(B) > 1e-15 else 0.0

    q_new = q_i + dq
    if q_new <= 0:
        q_new = q_i * 0.5
    return q_new, offset(q_new, h, lmd)


# ═══════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RayResult:
    """Result of a single two-point ray trace.

    Attributes
    ----------
    travel_time : float
        Total travel time (s).
    ray_path : numpy.ndarray
        Ray coordinates in the 2-D ray plane, shape ``(M, 2)``
        with columns ``[x, z]``.
    ray_parameter : float
        Horizontal slowness *p* (s/m).
    tstar : float or None
        Attenuation operator :math:`t^*` (s), if requested.
    spreading : float or None
        Geometrical spreading factor :math:`L`, if requested.
    trans_product : float or None
        Product of transmission-coefficient magnitudes along the ray,
        if requested.
    """

    travel_time: float
    ray_path: np.ndarray
    ray_parameter: float
    tstar: float | None = None
    spreading: float | None = None
    trans_product: float | None = None


# ═══════════════════════════════════════════════════════════════════════
#  Main solver
# ═══════════════════════════════════════════════════════════════════════

def solve(
    stack: LayerStack,
    epicentral_dist: float,
    z_src: float,
    z_rcv: float,
    vel_type: str = "Vp",
    compute_amplitude: bool = False,
    transcoef_method: str = "angle",
    tol: float = 1e-4,
    max_iter: int = 10,
) -> RayResult:
    r"""Solve the two-point ray tracing problem for a single S–R pair.

    Algorithm
    ---------
    1.  Build dimensionless parameters :math:`\lambda_k = v_k/v_{\max}`.
    2.  Compute initial :math:`q_0` via :func:`initial_q`.
    3.  Iterate :func:`newton_step` until
        :math:`|X(q)-X_R|<\mathtt{tol}`.
    4.  Convert :math:`q \to p`, propagate ray layer-by-layer.
    5.  Optionally compute :math:`t^*`, spreading and transmission
        inline.

    Parameters
    ----------
    stack : LayerStack
        Layers between source and receiver (from :func:`build_layer_stack`).
    epicentral_dist : float
        Horizontal distance (m) between source and receiver.
    z_src, z_rcv : float
        Source and receiver depths (m, positive downward).
    vel_type : str
        ``'Vp'`` or ``'Vs'``.
    compute_amplitude : bool
        If *True*, compute :math:`t^*`, geometrical spreading, and
        transmission coefficients alongside the travel time.
    transcoef_method : str
        ``'normal'`` or ``'angle'``.
    tol : float
        Convergence tolerance on the offset residual (m).
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    RayResult
    """
    v = stack.v(vel_type)
    h = stack.h
    N = stack.n_layers
    going_down = z_src <= z_rcv

    # ── Vertical / zero-offset ray ──
    if epicentral_dist < 1e-10:
        tt = float(np.sum(h / v))
        # Build vertical ray path
        pts = np.zeros((N + 1, 2))
        z = z_src
        for k in range(N):
            dz = h[k] if going_down else -h[k]
            z += dz
            pts[k + 1, 1] = z

        tstar = None
        if compute_amplitude:
            Q = stack.q_factor(vel_type)
            tstar = float(np.sum(h / v / Q)) if Q is not None else None

        return RayResult(
            travel_time=tt,
            ray_path=pts,
            ray_parameter=0.0,
            tstar=tstar,
            spreading=None,  # undefined for vertical ray (p=0)
            trans_product=None,
        )

    # ── Same-layer (direct line) ──
    if N == 1:
        dz = abs(z_rcv - z_src)
        dist = np.sqrt(epicentral_dist ** 2 + dz ** 2)
        tt = dist / v[0]
        p = epicentral_dist / (v[0] * dist)
        pts = np.array([[0.0, z_src], [epicentral_dist, z_rcv]])

        tstar = None
        trans_prod = None
        spreading = None
        if compute_amplitude:
            Q = stack.q_factor(vel_type)
            tstar = tt / Q[0] if Q is not None else None
            spreading = dist  # homogeneous spreading = distance
            trans_prod = 1.0  # no interfaces

        return RayResult(tt, pts, p, tstar, spreading, trans_prod)

    # ── General multi-layer case ──
    vmax = float(np.max(v))
    lmd = v / vmax

    # 1. Initial estimate
    q = initial_q(epicentral_dist, h, lmd)

    # 2. Newton iteration
    converged = False
    for _ in range(max_iter):
        q, X_new = newton_step(q, epicentral_dist, h, lmd)
        if abs(X_new - epicentral_dist) < tol:
            converged = True
            break

    # 3. Fallback to bounded minimisation
    if not converged:
        def _residual(qq):
            return (offset(qq, h, lmd) - epicentral_dist) ** 2

        res = opt.minimize_scalar(_residual, bounds=(1e-10, 1e8), method="bounded")
        q = res.x

    # 4. Convert to ray parameter
    p = p_from_q(q, vmax)

    # ── Propagate ray ──
    tt = 0.0
    tstar_val = 0.0
    trans_prod_val = 1.0
    Q_arr = stack.q_factor(vel_type) if compute_amplitude else None

    pts = np.zeros((N + 1, 2))
    pts[0] = [0.0, z_src]
    x_cum = 0.0
    z_cum = z_src

    for k in range(N):
        eta_k = np.sqrt(1.0 / (v[k] ** 2) - p ** 2)
        dx_k = h[k] * p / eta_k
        dt_k = h[k] / (v[k] ** 2 * eta_k)  # = h / (v² η)

        tt += dt_k
        x_cum += dx_k
        z_cum += h[k] if going_down else -h[k]
        pts[k + 1] = [x_cum, z_cum]

        # inline t*
        if compute_amplitude and Q_arr is not None:
            tstar_val += dt_k / Q_arr[k]

        # transmission coefficient at top of this layer (interface k)
        if compute_amplitude and k > 0:
            trans_prod_val *= _interface_transmission(
                p, k - 1, k, stack, vel_type, transcoef_method
            )

    # ── Geometrical spreading ──
    spreading_val = None
    if compute_amplitude:
        dXdq = offset_dq(q, h, lmd)
        pv = p * vmax
        denom_dp = (1.0 - pv * pv)
        if denom_dp > 1e-15:
            dqdp = vmax / denom_dp ** 1.5
            dXdp = dXdq * dqdp

            cos_is = np.sqrt(max(1.0 - (p * v[0]) ** 2, 0.0))
            cos_ir = np.sqrt(max(1.0 - (p * v[-1]) ** 2, 0.0))

            denom = cos_is * cos_ir
            if denom > 1e-15 and abs(dXdp) > 0:
                spreading_val = float(
                    np.sqrt(epicentral_dist * abs(dXdp) / denom)
                )

    return RayResult(
        travel_time=tt,
        ray_path=pts,
        ray_parameter=p,
        tstar=tstar_val if compute_amplitude and Q_arr is not None else None,
        spreading=spreading_val,
        trans_product=trans_prod_val if compute_amplitude else None,
    )


# ── Transmission helper ──

def _interface_transmission(
    p: float,
    k_above: int,
    k_below: int,
    stack: LayerStack,
    vel_type: str,
    method: str,
) -> float:
    """Transmission coefficient at the interface between layers *k_above* and *k_below*."""
    from .amplitude import transmission_normal, transmission_psv

    v_above = stack.v(vel_type)[k_above]
    v_below = stack.v(vel_type)[k_below]

    if stack.rho is None:
        return 1.0  # can't compute without density

    rho_above = stack.rho[k_above]
    rho_below = stack.rho[k_below]

    if method == "normal":
        return abs(transmission_normal(v_above, rho_above, v_below, rho_below))

    # Angle-dependent (full Zoeppritz)
    vp_a, vs_a = stack.vp[k_above], stack.vs[k_above]
    vp_b, vs_b = stack.vp[k_below], stack.vs[k_below]
    T = transmission_psv(p, vp_a, vs_a, rho_above, vp_b, vs_b, rho_below, vel_type)
    return abs(T)
