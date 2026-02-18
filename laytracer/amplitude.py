r"""
Amplitude-related quantities for ray-theory seismograms.

*   Normal-incidence and angle-dependent (Zoeppritz) transmission
    coefficients
*   Geometrical spreading from the ray-tube Jacobian
*   Attenuation operator :math:`t^*`

References
----------
.. footbibliography::
    AkiRichards2002
    LayWallace1995
    Cerveny2001
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
