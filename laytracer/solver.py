r"""
Core two-point ray tracing solver using the dimensionless *q*-parameter.

Implements the method of :cite:t:`FangChen2019`:

*   Dimensionless ray parameter :math:`q` for numerical stability
*   Vectorised offset equation :math:`X(q)` and its derivatives
*   Asymptotic initial estimate (Section 2.2 of the paper)
*   Second-order Newton iteration (Section 2.3 of the paper)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize as opt

from .model import LayerStack, normalize_phase, q_factor_key

_FLOAT_EPS = np.finfo(float).eps

SOLVE_OUTPUTS = frozenset({
    "travel_times",
    "rays",
    "ray_parameters",
    "tstar",
    "spreading",
    "trans_product",
    "complex_coefficient_product",
    "diagnostics",
})


def _normalize_requested(requested):
    if requested is None:
        return None

    normalized = frozenset(str(name) for name in requested)
    invalid = normalized - SOLVE_OUTPUTS
    if invalid:
        valid = ", ".join(sorted(SOLVE_OUTPUTS))
        invalid_str = ", ".join(sorted(invalid))
        raise ValueError(f"Invalid requested outputs: {invalid_str}. Valid outputs: {valid}")

    if "travel_times" not in normalized:
        raise ValueError("requested must include 'travel_times'")

    return normalized


def _psv_key_phase(phase: str) -> str:
    """Return the one-letter key used by P-SV coefficient dictionaries."""
    return "p" if normalize_phase(phase) == "P" else "s"


def _normalize_transcoef_method(method: str) -> str:
    """Return the canonical interface-coefficient convention.

    ``"angle"`` was historically accepted as an undocumented spelling for
    the unnormalised (standard Zoeppritz) convention.  Keeping the alias is
    important for existing callers while making invalid values explicit.
    """
    method = str(method).lower()
    if method == "angle":
        return "standard"
    if method not in {"standard", "normalized"}:
        raise ValueError(
            "transcoef_method must be 'standard', 'normalized', or the "
            "legacy alias 'angle'."
        )
    return method


# ═══════════════════════════════════════════════════════════════════════
#  Offset equation and derivatives
# ═══════════════════════════════════════════════════════════════════════

def _offset_from_s(
    q: float,
    h: np.ndarray,
    lmd: np.ndarray,
    s: np.ndarray,
) -> float:
    """Evaluate the offset with model-only terms already computed."""
    return float(np.sum(q * lmd * h / np.sqrt(1.0 + s * q * q)))


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
    return _offset_from_s(q, h, lmd, s)


def _offset_dq_from_s(
    q: float,
    h: np.ndarray,
    lmd: np.ndarray,
    s: np.ndarray,
) -> float:
    """Evaluate the first derivative with model-only terms precomputed."""
    return float(np.sum(lmd * h / (1.0 + s * q * q) ** 1.5))


def offset_dq(q: float, h: np.ndarray, lmd: np.ndarray) -> float:
    r"""First derivative :math:`\mathrm{d}X/\mathrm{d}q`.

    .. math::
        \frac{\mathrm{d}X}{\mathrm{d}q}
        = \sum_{k=1}^{N}
          \frac{\lambda_k\,h_k}
               {\bigl[1 + (1 - \lambda_k^2)\,q^2\bigr]^{3/2}}
    """
    s = 1.0 - lmd * lmd
    return _offset_dq_from_s(q, h, lmd, s)


def _offset_dq2_from_s(
    q: float,
    h: np.ndarray,
    lmd: np.ndarray,
    s: np.ndarray,
) -> float:
    """Evaluate the second derivative with model-only terms precomputed."""
    return float(-3.0 * q * np.sum(s * lmd * h / (1.0 + s * q * q) ** 2.5))


def offset_dq2(q: float, h: np.ndarray, lmd: np.ndarray) -> float:
    r"""Second derivative :math:`\mathrm{d}^2X/\mathrm{d}q^2`.

    .. math::
        \frac{\mathrm{d}^2X}{\mathrm{d}q^2}
        = -3\,q\,\sum_{k=1}^{N}
          \frac{(1 - \lambda_k^2)\,\lambda_k\,h_k}
               {\bigl[1 + (1 - \lambda_k^2)\,q^2\bigr]^{5/2}}
    """
    s = 1.0 - lmd * lmd
    return _offset_dq2_from_s(q, h, lmd, s)


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
#  Initial estimate
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
#  Second-order Newton iteration
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
    s = 1.0 - lmd * lmd
    return _newton_step_from_s(q_i, X_target, h, lmd, s)


def _newton_step_from_s(
    q_i: float,
    X_target: float,
    h: np.ndarray,
    lmd: np.ndarray,
    s: np.ndarray,
) -> tuple[float, float]:
    """Perform one quadratic update using cached model-only terms."""
    X_i = _offset_from_s(q_i, h, lmd, s)
    C = X_i - X_target
    B = _offset_dq_from_s(q_i, h, lmd, s)
    A = 0.5 * _offset_dq2_from_s(q_i, h, lmd, s)

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
                e1 = abs(_offset_from_s(q1, h, lmd, s) - X_target)
                e2 = abs(_offset_from_s(q2, h, lmd, s) - X_target)
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
    return q_new, _offset_from_s(q_new, h, lmd, s)


# ═══════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SolveDiagnostics:
    r"""Numerical certificate for one two-point solve.

    Attributes
    ----------
    converged : bool
        Whether the independently recomputed offset meets ``tol``.
    method : str
        ``"q_newton"`` for the quadratic iteration or ``"q_brentq"`` for
        the safeguarded bracketed fallback.
    iterations : int
        Number of quadratic Newton updates attempted.
    fallback_iterations : int
        Number of bracketed iterations, or zero when no fallback was used.
    initial_q, final_q : float
        Initial and accepted dimensionless ray parameters.
    initial_ray_parameter, ray_parameter : float
        Initial and accepted physical horizontal slownesses (s/m).
    signed_offset_residual, absolute_offset_residual : float
        Recomputed offset minus requested offset, and its magnitude (m).
    conditioning : float
        Dimensionless logarithmic slope ``abs(q / X * dX/dq)``.  The
        zero-offset limit is reported as one.
    criticality_margin : float
        ``1 - p * vmax``; values approaching zero indicate grazing
        propagation in the fastest traversed leg.
    failure_reason : str or None
        Populated only when a numerical certificate cannot be produced.
    """

    converged: bool
    method: str
    iterations: int
    initial_q: float
    final_q: float
    initial_ray_parameter: float
    ray_parameter: float
    signed_offset_residual: float
    absolute_offset_residual: float
    conditioning: float
    criticality_margin: float
    fallback_iterations: int = 0
    failure_reason: str | None = None


@dataclass
class RayResult:
    r"""Result of a single two-point ray trace.

    Attributes
    ----------
    travel_time : float
        Total travel time (s).
    ray_path : numpy.ndarray or None
        Ray coordinates in the 2-D ray plane, shape ``(M, 2)``
        with columns ``[x, z]``.  *None* if not requested.
    ray_parameter : float or None
        Horizontal slowness *p* (s/m).
    tstar : float or None
        Attenuation operator :math:`t^*` (s), if requested.
    spreading : float or None
        Relative geometrical spreading factor :math:`\mathcal{L}`, if requested.
    trans_product : float or None
        Product of transmission-coefficient magnitudes along the ray,
        if requested.
    complex_coefficient_product : complex or None
        Signed/complex cumulative interface coefficient, if requested.
    diagnostics : SolveDiagnostics or None
        Numerical certificate, if requested.
    """

    travel_time: float
    ray_path: np.ndarray | None
    ray_parameter: float | None
    tstar: float | None = None
    spreading: float | None = None
    trans_product: float | None = None
    complex_coefficient_product: complex | None = None
    diagnostics: SolveDiagnostics | None = None


def _validate_dynamic_inputs(
    segments: list[dict],
    interactions: list[dict],
    need_tstar: bool,
    need_coefficients: bool,
) -> None:
    """Validate only properties needed by explicitly requested attributes."""
    if need_tstar:
        for seg in segments:
            q_values = seg.get(q_factor_key(seg["phase"]))
            if q_values is None:
                raise ValueError(
                    f"{q_factor_key(seg['phase']).upper()} is required when tstar is requested."
                )
            q_values = np.asarray(q_values, dtype=float)
            if np.any(~np.isfinite(q_values)) or np.any(q_values <= 0.0):
                raise ValueError("Requested tstar requires finite, positive Q values.")

    if not need_coefficients:
        return

    interacting_segments = {int(inter["seg_idx"]) for inter in interactions}
    for seg_i, seg in enumerate(segments):
        crosses_interface = len(seg["h"]) > 1 or seg_i in interacting_segments
        if not crosses_interface:
            continue
        rho = seg.get("rho")
        if rho is None:
            raise ValueError(
                "Rho is required when interface coefficients are requested."
            )
        rho = np.asarray(rho, dtype=float)
        if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
            raise ValueError(
                "Requested interface coefficients require finite, positive density."
            )

    for inter in interactions:
        rho_beyond = float(inter.get("rho_beyond", 0.0))
        if not np.isfinite(rho_beyond) or rho_beyond <= 0.0:
            raise ValueError(
                "Rho is required on both sides of every requested interaction coefficient."
            )


# ═══════════════════════════════════════════════════════════════════════
#  Main solver
# ═══════════════════════════════════════════════════════════════════════

def solve(
    h: np.ndarray,
    v: np.ndarray,
    segments: list[dict],
    interactions: list[dict],
    epicentral_dist: float,
    z_src: float,
    z_rcv: float,
    requested=None,
    return_ray_path: bool = True,
    need_ray_parameter: bool = True,
    need_tstar: bool = False,
    need_spreading: bool = False,
    need_trans_product: bool = False,
    transcoef_method: str = "standard",
    tol: float = 1e-4,
    max_iter: int = 10,
    need_complex_coefficient_product: bool = False,
    need_diagnostics: bool = False,
) -> RayResult:
    r"""Solve the two-point ray tracing problem for an arbitrary path.

    Parameters
    ----------
    h : numpy.ndarray
        Concatenated layer thicknesses for the entire path (m).
    v : numpy.ndarray
        Concatenated phase velocities (Vp or Vs) for the entire path (m/s).
    segments : list of dict
        Metadata for each logical monotonic segment (used for reconstruction).
        Dict keys: 'h', 'v', 'phase', 'start_z', 'end_z'.
    interactions : list of dict
        Metadata for interactions (reflections/refractions) impacting amplitude.
        Dict keys: 'type', 'depth', 'in_phase', 'out_phase', 'seg_idx'.
    interactions are assumed to occur at the END of the defined segment.
    """
    requested = _normalize_requested(requested)
    if requested is not None:
        return_ray_path = "rays" in requested
        need_ray_parameter = "ray_parameters" in requested
        need_tstar = "tstar" in requested
        need_spreading = "spreading" in requested
        need_trans_product = "trans_product" in requested
        need_complex_coefficient_product = "complex_coefficient_product" in requested
        need_diagnostics = "diagnostics" in requested

    need_coefficients = need_trans_product or need_complex_coefficient_product
    if need_coefficients:
        transcoef_method = _normalize_transcoef_method(transcoef_method)
    if need_tstar or need_coefficients:
        _validate_dynamic_inputs(
            segments,
            interactions,
            need_tstar=need_tstar,
            need_coefficients=need_coefficients,
        )

    N = len(h)

    # ── Vertical / zero-offset ray ──
    if epicentral_dist < 1e-10:
        tt = float(np.sum(h / v))
        pts = None
        if return_ray_path:
            pts_list = [[0.0, z_src]]
            for seg in segments:
                if len(seg["h"]) > 0:
                    pts_list.append([0.0, seg["end_z"]])
            pts = np.array(pts_list)

        tstar = None
        trans_prod_val = 1.0 if need_trans_product else None
        complex_prod_val = 1.0 + 0.0j if need_complex_coefficient_product else None
        spreading_val = None
        
        if need_tstar or need_spreading or need_coefficients:
            # Use the p -> 0 limit of the direct-wave spreading expression.
            # For a layered vertical ray this reduces to sum(h_k * v_k), i.e.
            # Vrms^2 * travel time along the traversed path.
            tstar_val = 0.0
            spreading_val = float(np.sum(h * v)) if need_spreading else None
            
            # Since p=0 for vertical ray
            p_vert = 0.0
                                   
            for seg_i, seg in enumerate(segments):
                ph = seg["phase"]
                q_key = q_factor_key(ph)
                if need_tstar:
                    tstar_val += np.sum(seg["h"] / seg["v"] / seg[q_key])

                if need_coefficients:
                    n_lay = len(seg["h"])
                    z_start = seg["start_z"]
                    z_end = seg["end_z"]
                    going_down = z_end >= z_start
                    range_k = range(n_lay) if going_down else range(n_lay - 1, -1, -1)

                    for k in range_k:
                        has_next_layer = (k < n_lay - 1) if going_down else (k > 0)
                        if has_next_layer:
                            k_next = (k + 1) if going_down else (k - 1)
                            coeff = _calc_intra_transmission(
                                p_vert, k, k_next, seg, transcoef_method,
                                magnitude=False,
                            )
                            if need_trans_product:
                                trans_prod_val *= abs(coeff)
                            if need_complex_coefficient_product:
                                complex_prod_val *= coeff

                    for inter in interactions:
                        if inter["seg_idx"] == seg_i:
                            coeff = _calc_interaction_coeff(
                                p_vert, inter, segments, seg_i, transcoef_method,
                                magnitude=False,
                            )
                            if need_trans_product:
                                trans_prod_val *= abs(coeff)
                            if need_complex_coefficient_product:
                                complex_prod_val *= coeff

            tstar = float(tstar_val) if need_tstar else None

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

        return RayResult(
            travel_time=tt,
            ray_path=pts,
            ray_parameter=0.0 if need_ray_parameter else None,
            tstar=tstar,
            spreading=spreading_val,
            trans_product=trans_prod_val,
            complex_coefficient_product=complex_prod_val,
            diagnostics=diagnostics,
        )

    # ── Same-layer (direct line) ──
    if N == 1:
        # Simple straight line in one uniform block
        # dZ is total vertical distance traversed
        dz = np.sum(h) 
        dist = np.sqrt(epicentral_dist ** 2 + dz ** 2)
        tt = dist / v[0]
        p = epicentral_dist / (v[0] * dist)
        pts = np.array([[0.0, z_src], [epicentral_dist, z_rcv]]) if return_ray_path else None

        tstar = None
        trans_prod = None
        complex_prod = None
        spreading = None
        if need_tstar or need_spreading or need_coefficients:
             seg = segments[0]
             q_key = q_factor_key(seg["phase"])
             if need_tstar:
                 tstar = tt / seg[q_key][0]
             if need_spreading:
                 spreading = dist * v[-1]
             if need_trans_product:
                 trans_prod = 1.0
             if need_complex_coefficient_product:
                 complex_prod = 1.0 + 0.0j

        pv = float(p * v[0])
        q_direct = q_from_p(p, float(v[0])) if pv < 1.0 else np.inf
        diagnostics = SolveDiagnostics(
            converged=True,
            method="q_newton",
            iterations=0,
            initial_q=float(q_direct),
            final_q=float(q_direct),
            initial_ray_parameter=float(p),
            ray_parameter=float(p),
            signed_offset_residual=0.0,
            absolute_offset_residual=0.0,
            conditioning=0.0 if not np.isfinite(q_direct) else 1.0,
            criticality_margin=max(0.0, 1.0 - pv),
        ) if need_diagnostics else None

        return RayResult(
            travel_time=tt,
            ray_path=pts,
            ray_parameter=p if need_ray_parameter else None,
            tstar=tstar,
            spreading=spreading,
            trans_product=trans_prod,
            complex_coefficient_product=complex_prod,
            diagnostics=diagnostics,
        )

    # ── General multi-layer case ──
    vmax = float(np.max(v))
    lmd = v / vmax
    offset_s = 1.0 - lmd * lmd

    # 1. Initial estimate
    q = initial_q(epicentral_dist, h, lmd)
    q_initial = float(q) if need_diagnostics else None

    # 2. Newton iteration
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        q, X_new = _newton_step_from_s(
            q, epicentral_dist, h, lmd, offset_s
        )
        if not math.isfinite(q) or q <= 0.0 or not math.isfinite(X_new):
            break
        if abs(X_new - epicentral_dist) < tol:
            converged = True
            break

    # 3. Safeguarded fallback.  X(q) is monotone and X(0)=0, while the
    # fastest traversed leg guarantees X(q)->infinity as q->infinity.
    method = "q_newton" if need_diagnostics else None
    fallback_iterations = 0
    if not converged:
        if need_diagnostics:
            method = "q_brentq"

        def _signed_residual(qq):
            return _offset_from_s(qq, h, lmd, offset_s) - epicentral_dist

        q_hi = max(1.0, float(q) if math.isfinite(q) and q > 0.0 else 1.0)
        for _ in range(128):
            f_hi = _signed_residual(q_hi)
            if math.isfinite(f_hi) and f_hi >= 0.0:
                break
            q_hi *= 2.0
        else:
            raise RuntimeError("Could not bracket the transformed ray-parameter root.")

        if need_diagnostics:
            q, fallback_result = opt.brentq(
                _signed_residual,
                0.0,
                q_hi,
                xtol=1e-14,
                rtol=4.0 * _FLOAT_EPS,
                maxiter=200,
                full_output=True,
                disp=False,
            )
            fallback_iterations = int(fallback_result.iterations)
        else:
            q = opt.brentq(
                _signed_residual,
                0.0,
                q_hi,
                xtol=1e-14,
                rtol=4.0 * _FLOAT_EPS,
                maxiter=200,
                disp=False,
            )

    # 4. Convert to ray parameter
    p = p_from_q(q, vmax)
    signed_residual = _offset_from_s(q, h, lmd, offset_s) - epicentral_dist
    roundoff_tol = 64.0 * _FLOAT_EPS * (
        epicentral_dist if epicentral_dist > 1.0 else 1.0
    )
    verification_tol = tol if tol >= roundoff_tol else roundoff_tol
    if not math.isfinite(signed_residual) or abs(signed_residual) > verification_tol:
        raise RuntimeError(
            "Two-point ray solve did not meet the requested offset tolerance: "
            f"residual={signed_residual:.6g} m, tolerance={verification_tol:.6g} m."
        )

    diagnostics = None
    if need_diagnostics:
        conditioning = abs(
            q * _offset_dq_from_s(q, h, lmd, offset_s) / epicentral_dist
        )
        diagnostics = SolveDiagnostics(
            converged=True,
            method=method,
            iterations=iterations,
            initial_q=q_initial,
            final_q=float(q),
            initial_ray_parameter=float(p_from_q(q_initial, vmax)),
            ray_parameter=float(p),
            signed_offset_residual=float(signed_residual),
            absolute_offset_residual=float(abs(signed_residual)),
            conditioning=float(conditioning),
            criticality_margin=float(max(0.0, 1.0 - p * vmax)),
            fallback_iterations=fallback_iterations,
        )

    # ── Propagate ray ──
    tt = 0.0
    tstar_val = 0.0
    trans_prod_val = 1.0 if need_trans_product else None
    complex_prod_val = 1.0 + 0.0j if need_complex_coefficient_product else None

    pts_list = [[0.0, z_src]] if return_ray_path else None
    x_cum = 0.0
    z_cum = z_src
    
    # We iterate over SEGMENTS to reconstruct the path logic
    # The solver treated 'h' and 'v' as one giant array.
    # We need to map back to the segments structure for Q and coordinates.
    
    for seg_i, seg in enumerate(segments):
        n_lay = len(seg["h"])
        seg_h = seg["h"]
        seg_v = seg["v"]
        
        # Amplitude stuff for this segment
        seg_q = seg[q_factor_key(seg["phase"])]
        # Determine direction for Z update
        start_z = seg["start_z"]
        end_z = seg["end_z"]
        going_down = end_z >= start_z
        
        # NOTE: segments["h"] is ordered top-to-bottom physically.
                
        range_k = range(n_lay) if going_down else range(n_lay - 1, -1, -1)
        
        for k in range_k:
                        
            val_v = seg_v[k]
            val_h = seg_h[k]
            
            eta_k = np.sqrt(1.0 / (val_v ** 2) - p ** 2)
            dx_k = val_h * p / eta_k
            dt_k = val_h / (val_v ** 2 * eta_k)
            
            tt += dt_k
            x_cum += dx_k
            
            # Update Z
            # If going down, z increases by h
            # If going up, z decreases by h
            dz = val_h if going_down else -val_h
            z_cum += dz
            
            if return_ray_path:
                pts_list.append([x_cum, z_cum])

            if need_tstar:
                tstar_val += dt_k / seg_q[k]
                
            # Transmission logic within the segment
            # This handles standard transmission across interfaces INSIDE the monotonic stack.
            # If we are traversing indices k -> k+1 (Down) or k -> k-1 (Up):
            # Interface is between them.
            if need_coefficients:
                # Identification of the interface
                has_next_layer = (k < n_lay - 1) if going_down else (k > 0)
                
                if has_next_layer:
                    # Index of the next layer
                    k_next = (k + 1) if going_down else (k - 1)
                    
                    coeff = _calc_intra_transmission(
                        p, k, k_next, seg, transcoef_method,
                        magnitude=False,
                    )
                    if need_trans_product:
                        trans_prod_val *= abs(coeff)
                    if need_complex_coefficient_product:
                        complex_prod_val *= coeff

        # ── Explicit Interaction at End of Segment ──
        # Check if there is an interaction defined for this segment index
        if need_coefficients:
            # Find interaction where seg_idx == seg_i
            # (There should be at most one per segment end)
            for inter in interactions:
                if inter["seg_idx"] == seg_i:
                    coeff = _calc_interaction_coeff(
                        p, inter, segments, seg_i, transcoef_method,
                        magnitude=False,
                    )
                    if need_trans_product:
                        trans_prod_val *= abs(coeff)
                    if need_complex_coefficient_product:
                        complex_prod_val *= coeff

    pts = np.array(pts_list) if return_ray_path else None

    # ── Geometrical spreading ──
    spreading_val = None
    if need_spreading:
        dXdq = offset_dq(q, h, lmd)
        pv = p * vmax
        denom_dp = (1.0 - pv * pv)
        if denom_dp > 1e-15:
            dqdp = vmax / denom_dp ** 1.5
            dXdp = dXdq * dqdp

            first_seg = segments[0]
            last_seg = segments[-1]
            first_down = first_seg["end_z"] >= first_seg["start_z"]
            last_down = last_seg["end_z"] >= last_seg["start_z"]
            v_sourceside = float(first_seg["v"][0 if first_down else -1])
            v_receiverside = float(last_seg["v"][-1 if last_down else 0])
            
            cos_is = np.sqrt(max(1.0 - (p * v_sourceside) ** 2, 0.0))
            cos_ir = np.sqrt(max(1.0 - (p * v_receiverside) ** 2, 0.0))

            if p > 1e-15 and abs(dXdp) > 0:
                # Relative geometrical spreading (Cerveny, 2001)
                spreading_val = float(
                    np.sqrt(epicentral_dist * abs(dXdp) * cos_is * cos_ir / p)
                )

    return RayResult(
        travel_time=tt,
        ray_path=pts,
        ray_parameter=p if need_ray_parameter else None,
        tstar=tstar_val if need_tstar else None,
        spreading=spreading_val,
        trans_product=trans_prod_val,
        complex_coefficient_product=complex_prod_val,
        diagnostics=diagnostics,
    )


def transmission_product(
    p: float,
    segments: list[dict],
    interactions: list[dict],
    transcoef_method: str = "standard",
) -> float:
    """Return the product of interface-coefficient magnitudes."""
    magnitude_product, _ = _coefficient_products(
        p, segments, interactions, transcoef_method
    )
    return magnitude_product


def complex_coefficient_product(
    p: float,
    segments: list[dict],
    interactions: list[dict],
    transcoef_method: str = "standard",
) -> complex:
    """Return the signed/complex cumulative interface coefficient."""
    _, complex_product = _coefficient_products(
        p, segments, interactions, transcoef_method
    )
    return complex_product


def _coefficient_products(
    p: float,
    segments: list[dict],
    interactions: list[dict],
    transcoef_method: str,
) -> tuple[float, complex]:
    method = _normalize_transcoef_method(transcoef_method)
    _validate_dynamic_inputs(
        segments,
        interactions,
        need_tstar=False,
        need_coefficients=True,
    )
    magnitude_product = 1.0
    complex_product = 1.0 + 0.0j

    for seg_i, seg in enumerate(segments):
        n_lay = len(seg["h"])
        z_start = seg["start_z"]
        z_end = seg["end_z"]
        going_down = z_end >= z_start
        range_k = range(n_lay) if going_down else range(n_lay - 1, -1, -1)

        for k in range_k:
            has_next_layer = (k < n_lay - 1) if going_down else (k > 0)
            if has_next_layer:
                k_next = (k + 1) if going_down else (k - 1)
                coeff = _calc_intra_transmission(
                    p, k, k_next, seg, method, magnitude=False
                )
                magnitude_product *= abs(coeff)
                complex_product *= coeff

        for inter in interactions:
            if inter["seg_idx"] == seg_i:
                coeff = _calc_interaction_coeff(
                    p, inter, segments, seg_i, method, magnitude=False
                )
                magnitude_product *= abs(coeff)
                complex_product *= coeff

    return float(magnitude_product), complex(complex_product)

def _calc_interaction_coeff(
    p, inter, segments, seg_idx, method, magnitude: bool = True,
):
    from .amplitude import psv_rt_coefficients, sh_rt_coefficients, normalize_rt_coefficient

    # Current segment (incident side)
    seg_in = segments[seg_idx]
    
    vp_in_arr = seg_in["vp"]
    vs_in_arr = seg_in["vs"]
    rho_in_arr = seg_in["rho"]
    
    # Determine direction of incident segment
    going_down = seg_in["end_z"] >= seg_in["start_z"]
    
    # Index of the layer touching the interface
    idx_in = -1 if going_down else 0
    
    vp1 = float(vp_in_arr[idx_in])
    vs1 = float(vs_in_arr[idx_in])
    rho1 = float(rho_in_arr[idx_in]) if rho_in_arr is not None else 0.0
    
    # Material properties of Transmission/Target medium
    vp2 = inter["vp_beyond"]
    vs2 = inter["vs_beyond"]
    rho2 = inter["rho_beyond"]
    
    if rho1 <= 0 or rho2 <= 0:
        raise ValueError("Positive density is required for interface coefficients.")
    
    in_phase = normalize_phase(inter["in_phase"])
    out_phase = normalize_phase(inter["out_phase"])
    itype = inter["type"]

    if "SH" in (in_phase, out_phase) and in_phase != out_phase:
        raise ValueError("SH is decoupled from P-SV in isotropic 1-D media.")

    prefix = "R" if itype == "reflection" else "T"
    if in_phase == "SH":
        RT = sh_rt_coefficients(p, vs1, rho1, vs2, rho2)
        key = f"{prefix}shsh"
        R_bar = RT.get(key, 0.0)
    else:
        key = f"{prefix}{_psv_key_phase(in_phase)}{_psv_key_phase(out_phase)}"
        RT = psv_rt_coefficients(p, vp1, vs1, rho1, vp2, vs2, rho2)
        R_bar = RT.get(key, 0.0)
    
    if method == "normalized":
        v_in = vp1 if in_phase == "P" else vs1
        v_out = vp2 if out_phase == "P" else vs2
        if itype == "reflection":
            v_out = vp1 if out_phase == "P" else vs1
            rho2 = rho1
        value = normalize_rt_coefficient(R_bar, p, v_in, rho1, v_out, rho2)
    else:
        value = R_bar

    return float(abs(value)) if magnitude else complex(value)


def _calc_intra_transmission(
    p: float,
    k_curr: int,
    k_next: int,
    seg: dict,
    method: str,
    magnitude: bool = True,
) -> float | complex:
    """Calculate transmission coefficient between two adjacent layers within a monotonic segment."""
    from .amplitude import psv_rt_coefficients, sh_rt_coefficients, normalize_rt_coefficient

    # Material properties
    vp1 = float(seg["vp"][k_curr])
    vs1 = float(seg["vs"][k_curr])
    rho1 = float(seg["rho"][k_curr])
    
    vp2 = float(seg["vp"][k_next])
    vs2 = float(seg["vs"][k_next])
    rho2 = float(seg["rho"][k_next])
    
    ph = normalize_phase(seg["phase"])
    if ph == "SH":
        RT = sh_rt_coefficients(p, vs1, rho1, vs2, rho2)
        R_bar = RT.get("Tshsh", 0.0)
    else:
        key = "Tpp" if ph == "P" else "Tss"
        RT = psv_rt_coefficients(p, vp1, vs1, rho1, vp2, vs2, rho2)
        R_bar = RT.get(key, 0.0)
    
    if method == "normalized":
        v_in = vp1 if ph == "P" else vs1
        v_out = vp2 if ph == "P" else vs2
        value = normalize_rt_coefficient(R_bar, p, v_in, rho1, v_out, rho2)
    else:
        value = R_bar

    return float(abs(value)) if magnitude else complex(value)



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
    from .amplitude import psv_rt_coefficients, sh_rt_coefficients, normalize_rt_coefficient

    v_above = stack.v(vel_type)[k_above]
    v_below = stack.v(vel_type)[k_below]

    if stack.rho is None:
        return 1.0  # can't compute without density

    rho_above = stack.rho[k_above]
    rho_below = stack.rho[k_below]

    vel_type_norm = str(vel_type).lower()
    if vel_type_norm in ("vp", "vs"):
        ph = "P" if vel_type_norm == "vp" else "SV"
    else:
        ph = normalize_phase(vel_type)
    vp_a, vs_a = stack.vp[k_above], stack.vs[k_above]
    vp_b, vs_b = stack.vp[k_below], stack.vs[k_below]
    if ph == "SH":
        RT = sh_rt_coefficients(p, vs_a, rho_above, vs_b, rho_below)
        R_bar = RT["Tshsh"]
    else:
        RT = psv_rt_coefficients(p, vp_a, vs_a, rho_above, vp_b, vs_b, rho_below)
        key = "Tpp" if ph == "P" else "Tss"
        R_bar = RT[key]

    if method == "normalized":
        v_in = float(v_above)
        v_out = float(v_below)
        return float(abs(normalize_rt_coefficient(R_bar, p, v_in, rho_above, v_out, rho_below)))

    return float(abs(R_bar))
