r"""
Velocity model representation and layer geometry for 1-D layered media.

This module provides a :class:`LayerStack` data structure that encapsulates
the sequence of layers traversed by a ray between a source and receiver,
and a helper :func:`build_layer_stack` to extract this structure from a
velocity model DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LayerStack:
    r"""Layers traversed by a ray between source and receiver depths.

    Parameters
    ----------
    h : numpy.ndarray
        Layer thicknesses (m), shape ``(N,)``.  Ordered from the
        shallowest traversed depth to the deepest.
    vp : numpy.ndarray
        P-wave velocities (m/s), shape ``(N,)``.
    vs : numpy.ndarray
        S-wave velocities (m/s), shape ``(N,)``.
    rho : numpy.ndarray or None
        Densities (kg/m³), shape ``(N,)``, or *None*.
    qp : numpy.ndarray or None
        P-wave quality factors, shape ``(N,)``, or *None*.
    qs : numpy.ndarray or None
        S-wave quality factors, shape ``(N,)``, or *None*.
    """

    h: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray | None = None
    qp: np.ndarray | None = None
    qs: np.ndarray | None = None

    @property
    def n_layers(self) -> int:
        """Number of layers in the stack."""
        return len(self.h)

    def v(self, vel_type: str = "Vp") -> np.ndarray:
        """Return the velocity array for the requested wave type.

        Parameters
        ----------
        vel_type : str
            ``'Vp'`` or ``'Vs'``.
        """
        if vel_type.lower() in ("vp", "p"):
            return self.vp
        elif vel_type.lower() in ("vs", "s"):
            return self.vs
        raise ValueError(f"vel_type must be 'Vp' or 'Vs', got '{vel_type}'")

    def q_factor(self, vel_type: str = "Vp") -> np.ndarray | None:
        """Return the Q-factor array for the requested wave type."""
        if vel_type.lower() in ("vp", "p"):
            return self.qp
        return self.qs


def _layer_index(depth: float, boundaries: np.ndarray) -> int:
    """Return the index of the layer containing *depth*.

    Layer *k* spans from ``boundaries[k]`` (inclusive) to
    ``boundaries[k+1]`` (exclusive).  The last layer extends to
    infinity.

    Parameters
    ----------
    depth : float
        Query depth (positive downward).
    boundaries : numpy.ndarray
        Sorted layer-top depths from the velocity model.

    Returns
    -------
    int
        Layer index (0-based).
    """
    idx = int(np.searchsorted(boundaries, depth, side="right")) - 1
    return max(idx, 0)


def build_layer_stack(
    vel_df: pd.DataFrame,
    z_src: float,
    z_rcv: float,
) -> LayerStack:
    r"""Extract the layer stack between source and receiver depths.

    The returned :class:`LayerStack` contains the layers traversed by a
    ray connecting *z_src* and *z_rcv*, with partial thicknesses at the
    source and receiver layers.  Layers are always ordered from the
    shallowest point to the deepest, regardless of which endpoint is the
    source.

    Parameters
    ----------
    vel_df : pandas.DataFrame
        Velocity model.  Required columns: ``Depth``, ``Vp``, ``Vs``.
        Optional columns: ``Rho``, ``Qp``, ``Qs``.
        ``Depth`` values define the *top* of each layer.
    z_src : float
        Source depth (positive downward, m).
    z_rcv : float
        Receiver depth (positive downward, m).

    Returns
    -------
    LayerStack
    """
    depths = vel_df["Depth"].values.astype(np.float64)
    vp_all = vel_df["Vp"].values.astype(np.float64)
    vs_all = vel_df["Vs"].values.astype(np.float64)

    has_rho = "Rho" in vel_df.columns
    has_qp = "Qp" in vel_df.columns
    has_qs = "Qs" in vel_df.columns

    rho_all = vel_df["Rho"].values.astype(np.float64) if has_rho else None
    qp_all = vel_df["Qp"].values.astype(np.float64) if has_qp else None
    qs_all = vel_df["Qs"].values.astype(np.float64) if has_qs else None

    z_top = min(z_src, z_rcv)
    z_bot = max(z_src, z_rcv)

    i_top = _layer_index(z_top, depths)
    i_bot = _layer_index(z_bot, depths)

    # ── Single-layer case ──
    if i_top == i_bot:
        h_arr = np.array([z_bot - z_top])
        slc = slice(i_top, i_top + 1)
        return LayerStack(
            h=h_arr,
            vp=vp_all[slc].copy(),
            vs=vs_all[slc].copy(),
            rho=rho_all[slc].copy() if has_rho else None,
            qp=qp_all[slc].copy() if has_qp else None,
            qs=qs_all[slc].copy() if has_qs else None,
        )

    # ── Multi-layer case ──
    n = i_bot - i_top + 1
    h_arr = np.empty(n)

    # First (shallowest) layer — partial
    if i_top + 1 < len(depths):
        h_arr[0] = depths[i_top + 1] - z_top
    else:
        h_arr[0] = z_bot - z_top  # only layer

    # Middle layers — full thickness
    for k in range(1, n - 1):
        idx = i_top + k
        if idx + 1 < len(depths):
            h_arr[k] = depths[idx + 1] - depths[idx]
        else:
            h_arr[k] = z_bot - depths[idx]

    # Last (deepest) layer — partial
    h_arr[n - 1] = z_bot - depths[i_bot]

    slc = slice(i_top, i_bot + 1)
    return LayerStack(
        h=h_arr,
        vp=vp_all[slc].copy(),
        vs=vs_all[slc].copy(),
        rho=rho_all[slc].copy() if has_rho else None,
        qp=qp_all[slc].copy() if has_qp else None,
        qs=qs_all[slc].copy() if has_qs else None,
    )
