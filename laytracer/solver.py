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
    h: np.ndarray,
    v: np.ndarray,
    segments: list[dict],
    interactions: list[dict],
    epicentral_dist: float,
    z_src: float,
    z_rcv: float,
    compute_amplitude: bool = False,
    transcoef_method: str = "angle",
    tol: float = 1e-4,
    max_iter: int = 10,
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
    N = len(h)    

    # ── Vertical / zero-offset ray ──
    if epicentral_dist < 1e-10:
        tt = float(np.sum(h / v))
        # Build vertical ray path
        # Reconstruct path based on segments
        # This is a bit complex for vertical rays with bounces, but solvable.
        pts_list = []
        curr_x = 0.0
        # Start point
        pts_list.append([0.0, z_src])
        
        for seg in segments:
            seg_h = seg["h"]
            n_seg = len(seg_h)
            z_start = seg["start_z"]
            z_end = seg["end_z"]
            going_down = z_end >= z_start
            
            # For geometric path reconstruction
            # We just need to know the sequence of depths
            # Vertical ray: x is constant 0
            
            # Simple Z reconstruction:
            # We assume segments are continuous.
            if n_seg > 0:
                # Add end point of this segment
                pts_list.append([0.0, z_end])
        
        pts = np.array(pts_list)

        tstar = None
        trans_prod_val = None
        
        if compute_amplitude:
            # sum dt / Q over all layers
            tstar_val = 0.0
            trans_prod_val = 1.0
            
            # Since p=0 for vertical ray
            p_vert = 0.0
                                   
            for seg_i, seg in enumerate(segments):
                ph = seg["phase"]
                q_key = "qp" if ph == "P" else "qs"
                # Handle missing Q
                if seg[q_key] is not None:
                     tstar_val += np.sum(seg["h"] / seg["v"] / seg[q_key])
                     
                # Transmission within segment
                n_lay = len(seg["h"])
                # Direction doesn't change physics of transmission coeff formula for p=0 much
                # but we need correct k_curr, k_next.
                z_start = seg["start_z"]
                z_end = seg["end_z"]
                going_down = z_end >= z_start
                
                # Intra-segment transmission
                range_k = range(n_lay) if going_down else range(n_lay - 1, -1, -1)
                
                for k in range_k:
                    # Check for next layer
                    has_next_layer = (k < n_lay - 1) if going_down else (k > 0)
                    if has_next_layer:
                        k_next = (k + 1) if going_down else (k - 1)
                        if seg["rho"] is not None:
                            trans_prod_val *= _calc_intra_transmission(
                                p_vert, k, k_next, seg, transcoef_method
                            )
                            
                # Explicit interaction at end of segment
                for inter in interactions:
                    if inter["seg_idx"] == seg_i:
                        coeff = _calc_interaction_coeff(
                            p_vert, inter, segments, seg_i, transcoef_method
                        )
                        trans_prod_val *= coeff

            tstar = float(tstar_val)

        return RayResult(
            travel_time=tt,
            ray_path=pts,
            ray_parameter=0.0,
            tstar=tstar,
            spreading=None,  # undefined for vertical ray (p=0)
            trans_product=trans_prod_val,
        )

    # ── Same-layer (direct line) ──
    if N == 1:
        # Simple straight line in one uniform block
        # dZ is total vertical distance traversed
        dz = np.sum(h) 
        dist = np.sqrt(epicentral_dist ** 2 + dz ** 2)
        tt = dist / v[0]
        p = epicentral_dist / (v[0] * dist)
        pts = np.array([[0.0, z_src], [epicentral_dist, z_rcv]])

        tstar = None
        trans_prod = None
        spreading = None
        if compute_amplitude:
             # Need Q from the single segment
             seg = segments[0]
             q_key = "qp" if seg["phase"] == "P" else "qs"
             if seg[q_key] is not None:
                 tstar = tt / seg[q_key][0]
             spreading = dist
             trans_prod = 1.0

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
    
    # Path reconstruction
    pts_list = [[0.0, z_src]]
    x_cum = 0.0
    z_cum = z_src
    
    # We iterate over SEGMENTS to reconstruct the path logic
    # The solver treated 'h' and 'v' as one giant array.
    # We need to map back to the segments structure for Q and coordinates.
    
    global_k = 0 # index into the flattened velocity/thickness arrays
    
    for seg_i, seg in enumerate(segments):
        n_lay = len(seg["h"])
        seg_h = seg["h"]
        seg_v = seg["v"]
        
        # Amplitude stuff for this segment
        seg_q = seg["qp"] if seg["phase"] == "P" else seg["qs"]
        # Determine direction for Z update
        start_z = seg["start_z"]
        end_z = seg["end_z"]
        going_down = end_z >= start_z
        
        # NOTE: segments["h"] is ordered top-to-bottom physically.
        # If we are going UP (end_z < start_z), we traverse slices in reverse order?
        # api.py logic: "build_layer_stack returns layers shallow->deep."
        # If going down: we traverse index 0 -> N
        # If going up: we traverse index N -> 0?
        # Let's check how api.py populated 'seg_v'.
        # api.py: "v_leg = leg_stack.v(...) ... h_total.append(leg_stack.h)"
        # This implies seg["v"] is also ordered shallow-to-deep.
        # So YES, if going UP, we must iterate 'k' backwards relative to the array.
        
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
            
            pts_list.append([x_cum, z_cum])
            
            if compute_amplitude and seg_q is not None:
                tstar_val += dt_k / seg_q[k]
                
            # Transmission logic within the segment
            # This handles standard transmission across interfaces INSIDE the monotonic stack.
            # If we are traversing indices k -> k+1 (Down) or k -> k-1 (Up):
            # Interface is between them.
            if compute_amplitude:
                # Identification of the interface
                has_next_layer = (k < n_lay - 1) if going_down else (k > 0)
                
                if has_next_layer:
                    # Index of the next layer
                    k_next = (k + 1) if going_down else (k - 1)
                    
                    # We need density to compute transmission
                    # If density is None, we assume T=1.0
                    if seg["rho"] is not None:
                        # Call helper
                        coeff = _calc_intra_transmission(
                            p, k, k_next, seg, transcoef_method
                        )
                        trans_prod_val *= coeff

        # ── Explicit Interaction at End of Segment ──
        # Check if there is an interaction defined for this segment index
        if compute_amplitude:
            # Find interaction where seg_idx == seg_i
            # (There should be at most one per segment end)
            for inter in interactions:
                if inter["seg_idx"] == seg_i:
                    coeff = _calc_interaction_coeff(p, inter, segments, seg_i, transcoef_method)
                    trans_prod_val *= coeff

    pts = np.array(pts_list)

    # ── Geometrical spreading ──
    spreading_val = None
    if compute_amplitude:
        dXdq = offset_dq(q, h, lmd)
        pv = p * vmax
        denom_dp = (1.0 - pv * pv)
        if denom_dp > 1e-15:
            dqdp = vmax / denom_dp ** 1.5
            dXdp = dXdq * dqdp

            # Geometric spreading depends on Source/Receiver velocities
            # v[0] and v[-1] in the flattened array correspond to start/end of path
            # IF we constructed it correctly.
            v_sourceside = v[0] 
            v_receiverside = v[-1]
            
            cos_is = np.sqrt(max(1.0 - (p * v_sourceside) ** 2, 0.0))
            cos_ir = np.sqrt(max(1.0 - (p * v_receiverside) ** 2, 0.0))

            if p > 1e-15 and abs(dXdp) > 0:
                spreading_val = float(
                    (1.0 / v_sourceside) * np.sqrt(epicentral_dist * abs(dXdp) * cos_is * cos_ir / p)
                )

    return RayResult(
        travel_time=tt,
        ray_path=pts,
        ray_parameter=p,
        tstar=tstar_val if compute_amplitude else None,
        spreading=spreading_val,
        trans_product=trans_prod_val if compute_amplitude else None,
    )

def _calc_interaction_coeff(p, inter, segments, seg_idx, method):
    from .amplitude import transmission_normal, psv_rt_coefficients
    
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
        return 1.0 # Density missing, cannot compute dynamic amplitude
    
    in_phase = inter["in_phase"]
    out_phase = inter["out_phase"]
    itype = inter["type"]
    
    # Construct Key for psv_rt_coefficients
    # "Rpp", "Rps", "Tpp", "Tps" etc.
    
    prefix = "R" if itype == "reflection" else "T"
    key = f"{prefix}{in_phase.lower()}{out_phase.lower()}"
    
    if method == "normal":
         # Normal incidence approx
         v1 = vp1 if in_phase == "P" else vs1
         v2 = vp2 if out_phase == "P" else vs2 
         # We fallback to standard acoustic logic for P-P / S-S.
         if in_phase != out_phase:
             return 0.0
         return abs(transmission_normal(v1, rho1, v2, rho2)) if itype == "transmission" else abs((v2*rho2 - v1*rho1)/(v2*rho2 + v1*rho1)) # approx
        
        
    RT = psv_rt_coefficients(p, vp1, vs1, rho1, vp2, vs2, rho2)
    
    # The keys in psv_rt_coefficients are:
    # Incident P: Rpp, Rps, Tpp, Tps
    # Incident S: Rsp, Rss, Tsp, Tss
    
    return float(abs(RT.get(key, 0.0)))


def _calc_intra_transmission(
    p: float,
    k_curr: int,
    k_next: int,
    seg: dict,
    method: str,
) -> float:
    """Calculate transmission coefficient between two adjacent layers within a monotonic segment."""
    from .amplitude import transmission_normal, psv_rt_coefficients
    
    # Material properties
    # Both layers are in the same segment arrays
    vp1 = float(seg["vp"][k_curr])
    vs1 = float(seg["vs"][k_curr])
    rho1 = float(seg["rho"][k_curr])
    
    vp2 = float(seg["vp"][k_next])
    vs2 = float(seg["vs"][k_next])
    rho2 = float(seg["rho"][k_next])
    
    # Phase
    ph = seg["phase"]
    key = "Tpp" if ph == "P" else "Tss"
    
    if method == "normal":
        v1 = vp1 if ph == "P" else vs1
        v2 = vp2 if ph == "P" else vs2
        return abs(transmission_normal(v1, rho1, v2, rho2))
        
    # Angle dependent
    RT = psv_rt_coefficients(p, vp1, vs1, rho1, vp2, vs2, rho2)
    return float(abs(RT.get(key, 0.0)))



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
    from .amplitude import transmission_normal, psv_rt_coefficients

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
    RT = psv_rt_coefficients(p, vp_a, vs_a, rho_above, vp_b, vs_b, rho_below)
    key = "Tpp" if vel_type.lower() in ("vp", "p") else "Tss"
    return float(abs(RT[key]))
