r"""
Amplitude-related quantities for ray-theory seismograms.

*   Normal-incidence and angle-dependent (Zoeppritz) transmission
    coefficients :cite:p:`LayWallace1995`
*   Geometrical spreading from the ray-tube Jacobian :cite:p:`Cerveny2001`
*   Attenuation operator (global absorbtion factor) :math:`t^*` :cite:p:`Cerveny2001`

References
----------
"""

from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Transmission coefficients
# ═══════════════════════════════════════════════════════════════════════

def transmission_normal(
    v1: float,
    rho1: float,
    v2: float,
    rho2: float,
) -> float:
    r"""Normal-incidence displacement transmission coefficient.

    .. math::
        T = \frac{2\,Z_1}{Z_1 + Z_2}, \qquad Z_i = \rho_i\,v_i

    Parameters
    ----------
    v1, rho1 : float
        Velocity (m/s) and density (kg/m³) on the incident side.
    v2, rho2 : float
        Velocity and density on the transmitted side.

    Returns
    -------
    float
        Displacement amplitude transmission coefficient.
    """
    Z1 = rho1 * v1
    Z2 = rho2 * v2
    denom = Z1 + Z2
    if abs(denom) < 1e-30:
        return 1.0
    return 2.0 * Z1 / denom


# ═══════════════════════════════════════════════════════════════════════
#  Full P-SV reflection / transmission matrix
# ═══════════════════════════════════════════════════════════════════════

def psv_rt_coefficients(
    p,
    vp1: float,
    vs1: float,
    rho1: float,
    vp2: float,
    vs2: float,
    rho2: float,
) -> dict:
    r"""Compute all eight P-SV reflection/transmission coefficients.

    Direct port of the Ammon MATLAB ``PSVRTmatrix`` function
    (Lay & Wallace / Aki & Richards formulation).

    For an incident P-wave the system unknowns are
    :math:`[R_{PP},\; R_{PS},\; T_{PP},\; T_{PS}]`.
    For an incident SV-wave the unknowns are
    :math:`[R_{SP},\; R_{SS},\; T_{SP},\; T_{SS}]`.

    Parameters
    ----------
    p : float or array_like
        Ray parameter (horizontal slowness, s/m).  Scalar or 1-D array.
    vp1, vs1, rho1 : float
        P-velocity, S-velocity, density of the *incident* medium.
    vp2, vs2, rho2 : float
        Same for the *transmitted* medium.

    Returns
    -------
    dict
        Keys ``'Rpp'``, ``'Rps'``, ``'Rss'``, ``'Rsp'``,
        ``'Tpp'``, ``'Tps'``, ``'Tss'``, ``'Tsp'``.
        Each value has the same shape as *p* (complex).
    """
    p = np.asarray(p, dtype=complex)
    csqrt = np.lib.scimath.sqrt

    # Vertical slownesses
    eta_a1 = csqrt(1.0 / (vp1 * vp1) - p * p)
    eta_a2 = csqrt(1.0 / (vp2 * vp2) - p * p)
    eta_b1 = csqrt(1.0 / (vs1 * vs1) - p * p)
    eta_b2 = csqrt(1.0 / (vs2 * vs2) - p * p)

    a = rho2 * (1.0 - 2.0 * vs2 * vs2 * p * p) - rho1 * (1.0 - 2.0 * vs1 * vs1 * p * p)
    b = rho2 * (1.0 - 2.0 * vs2 * vs2 * p * p) + 2.0 * rho1 * vs1 * vs1 * p * p
    c = rho1 * (1.0 - 2.0 * vs1 * vs1 * p * p) + 2.0 * rho2 * vs2 * vs2 * p * p
    d = 2.0 * (rho2 * vs2 * vs2 - rho1 * vs1 * vs1)

    E = b * eta_a1 + c * eta_a2
    F = b * eta_b1 + c * eta_b2
    G = a - d * eta_a1 * eta_b2
    H = a - d * eta_a2 * eta_b1

    D = E * F + G * H * p * p

    # At exactly critical or grazing angles D → 0, producing 0/0 (NaN).
    # Suppress the resulting numpy warnings; the NaN values are correct
    # (the coefficients are singular at those isolated ray parameters).
    with np.errstate(invalid="ignore", divide="ignore"):
        # --- Incident P ---
        Rpp =  ((b * eta_a1 - c * eta_a2) * F - (a + d * eta_a1 * eta_b2) * H * p * p) / D
        Rps = -(2.0 * eta_a1 * (a * b + d * c * eta_a2 * eta_b2) * p * (vp1 / vs1)) / D
        Tpp =  (2.0 * rho1 * eta_a1 * F * (vp1 / vp2)) / D
        Tps =  (2.0 * rho1 * eta_a1 * H * p * (vp1 / vs2)) / D

        # --- Incident SV ---
        Rss = -((b * eta_b1 - c * eta_b2) * E - (a + d * eta_a2 * eta_b1) * G * p * p) / D
        Rsp = -(2.0 * eta_b1 * (a * b + d * c * eta_a2 * eta_b2) * p * (vs1 / vp1)) / D
        Tss =  (2.0 * rho1 * eta_b1 * E * (vs1 / vs2)) / D
        Tsp = -(2.0 * rho1 * eta_b1 * G * p * (vs1 / vp2)) / D

    return dict(Rpp=Rpp, Rps=Rps, Rss=Rss, Rsp=Rsp,
                Tpp=Tpp, Tps=Tps, Tss=Tss, Tsp=Tsp)


# ═══════════════════════════════════════════════════════════════════════
#  Brewster-angle detection
# ═══════════════════════════════════════════════════════════════════════

def find_brewster_angles(
    rt_coefficients: dict,
    angles: np.ndarray,
    keys: list[str] | None = None,
    threshold: float = 0.05,
    order: int = 20,
) -> dict[str, list[float]]:
    r"""Find Brewster-like angles (deep minima) in R/T coefficient curves.

    A **Brewster angle** (by analogy with optics) is an incidence angle
    at which a reflection or transmission coefficient passes through
    zero or a deep minimum.  Unlike critical angles, which depend only
    on velocity ratios, Brewster angles depend on *all six* elastic
    parameters (Vp, Vs, ρ in both media) and arise from destructive
    interference between displacement potentials at the interface.

    Parameters
    ----------
    rt_coefficients : dict
        Output of :func:`psv_rt_coefficients` — each value is a 1-D
        array of complex coefficients.
    angles : array_like
        Incidence angles (degrees) corresponding to the ray-parameter
        samples used in *rt_coefficients*. Must have the same length
        as the coefficient arrays.
    keys : list of str, optional
        Which coefficient keys to search (e.g. ``['Rps', 'Rss']``).
        By default all eight keys are searched.
    threshold : float, optional
        Only report minima whose absolute value is below this value.
        Default 0.05.
    order : int, optional
        Half-window size passed to :func:`scipy.signal.argrelmin`
        for local-minimum detection.  Default 20.

    Returns
    -------
    dict[str, list[float]]
        Mapping from coefficient key to a list of Brewster angles
        (degrees).  Keys with no detected minima are omitted.
    """
    from scipy.signal import argrelmin as _argrelmin

    angles = np.asarray(angles)
    if keys is None:
        keys = list(rt_coefficients.keys())

    result: dict[str, list[float]] = {}
    for key in keys:
        vals = np.abs(rt_coefficients[key])
        idx = _argrelmin(vals, order=order)[0]
        brewster = [float(angles[i]) for i in idx if vals[i] < threshold]
        if brewster:
            result[key] = brewster
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Červený normalized displacement coefficients
# ═══════════════════════════════════════════════════════════════════════

def normalize_rt_coefficient(
    R_bar,
    p: float,
    v_in: float,
    rho_in: float,
    v_out: float,
    rho_out: float,
):
    r"""Apply Červený (2001) normalization to a displacement R/T coefficient.

    Eq. 5.3.10:

    .. math::
        R_{mn} = \bar{R}_{mn}
        \left(
          \frac{V(\tilde Q)\,\rho(\tilde Q)\,P(\tilde Q)}
               {V(Q)\,\rho(Q)\,P(Q)}
        \right)^{1/2}

    where :math:`P(Q) = (1 - V^2 p^2)^{1/2}`.

    Parameters
    ----------
    R_bar : complex or float
        Standard (unnormalized) displacement coefficient.
    p : float
        Ray parameter (horizontal slowness, s/m).
    v_in : float
        Phase velocity of the incident wave (m/s).
    rho_in : float
        Density of the incident medium (kg/m³).
    v_out : float
        Phase velocity of the outgoing (reflected/transmitted) wave (m/s).
    rho_out : float
        Density of the outgoing medium (kg/m³).

    Returns
    -------
    complex or float
        Normalized displacement coefficient.
    """
    csqrt = np.lib.scimath.sqrt
    p = np.asarray(p)
    P_in = csqrt(1.0 - v_in**2 * p**2)
    P_out = csqrt(1.0 - v_out**2 * p**2)
    denom = v_in * rho_in * P_in
    if p.ndim == 0:
        if abs(denom) < 1e-30:
            return R_bar
        factor = csqrt((v_out * rho_out * P_out) / denom)
        return R_bar * factor
    # Vectorised path
    factor = np.ones_like(R_bar, dtype=complex)
    mask = np.abs(denom) >= 1e-30
    factor[mask] = csqrt(
        (v_out * rho_out * P_out[mask]) / denom[mask]
    )
    return R_bar * factor
