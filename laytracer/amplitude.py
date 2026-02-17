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


def transmission_psv(
    p: float,
    vp1: float,
    vs1: float,
    rho1: float,
    vp2: float,
    vs2: float,
    rho2: float,
    vel_type: str = "Vp",
) -> complex:
    r"""Angle-dependent P-SV transmission coefficient (Zoeppritz).

    Solves the full 4×4 Zoeppritz system following
    :footcite:t:`AkiRichards2002` (Ch. 5, eqs. 5.40–5.42).

    For an incident P-wave the system unknowns are
    :math:`[R_{PP},\; R_{PS},\; T_{PP},\; T_{PS}]`.
    For an incident SV-wave the unknowns are
    :math:`[R_{SP},\; R_{SS},\; T_{SP},\; T_{SS}]`.

    Parameters
    ----------
    p : float
        Horizontal slowness (s/m).
    vp1, vs1, rho1 : float
        P-velocity, S-velocity, density of the incident medium.
    vp2, vs2, rho2 : float
        Same for the transmitted medium.
    vel_type : str
        ``'Vp'`` for P-to-P transmission, ``'Vs'`` for SV-to-SV
        transmission.

    Returns
    -------
    complex
        Displacement amplitude transmission coefficient.
    """
    # Vertical slownesses η = cos(θ)/v  (complex if post-critical)
    def _eta(v):
        arg = 1.0 / (v * v) - p * p
        if arg >= 0:
            return np.sqrt(arg)
        return 1j * np.sqrt(-arg)

    eta_p1 = _eta(vp1)
    eta_s1 = _eta(vs1)
    eta_p2 = _eta(vp2)
    eta_s2 = _eta(vs2)

    p2 = p * p

    # 4×4 system matrix (displacement + stress continuity)
    # Columns: [R_P, R_S, T_P, T_S]
    M = np.array([
        # Row 0: horizontal displacement (u_x)
        [-p,     -eta_s1,  p,      eta_s2],
        # Row 1: vertical displacement (u_z)
        [eta_p1, -p,       eta_p2, p     ],
        # Row 2: tangential stress (σ_xz)
        [2 * rho1 * vs1**2 * p * eta_p1,
         rho1 * vs1 * (1 - 2 * vs1**2 * p2) / vs1,
         2 * rho2 * vs2**2 * p * eta_p2,
         -rho2 * vs2 * (1 - 2 * vs2**2 * p2) / vs2],
        # Row 3: normal stress (σ_zz)
        [-rho1 * vp1 * (1 - 2 * vs1**2 * p2),
         2 * rho1 * vs1**2 * p * eta_s1,
         rho2 * vp2 * (1 - 2 * vs2**2 * p2),
         2 * rho2 * vs2**2 * p * eta_s2],
    ], dtype=complex)

    if vel_type.lower() in ("vp", "p"):
        # Incident P-wave RHS
        rhs = np.array([
            p,
            eta_p1,
            2 * rho1 * vs1**2 * p * eta_p1,
            rho1 * vp1 * (1 - 2 * vs1**2 * p2),
        ], dtype=complex)
    else:
        # Incident SV-wave RHS
        rhs = np.array([
            eta_s1,
            p,
            rho1 * vs1 * (1 - 2 * vs1**2 * p2) / vs1,
            -2 * rho1 * vs1**2 * p * eta_s1,
        ], dtype=complex)

    try:
        x = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return 1.0 + 0j

    # x = [R_P, R_S, T_P, T_S]
    if vel_type.lower() in ("vp", "p"):
        return complex(x[2])  # T_PP
    else:
        return complex(x[3])  # T_SS
