r"""
Standalone visualisation helpers for LayTracer.

Provides matplotlib-based 2-D plots and a Plotly-based 3-D ray viewer.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def velocity_profile(
    vel_df: pd.DataFrame,
    vel_type: str = "Vp",
    ax=None,
    color: str | None = None,
    label: str | None = None,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    **kwargs,
):
    r"""Plot a 1-D velocity–depth step profile.

    Parameters
    ----------
    vel_df : pandas.DataFrame
        Velocity model (columns ``Depth``, ``Vp``, ``Vs``).
    vel_type : str
        ``'Vp'`` or ``'Vs'``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  Created if *None*.
    color : str, optional
        Line colour.
    label : str, optional
        Legend label.
    xlim, ylim : tuple, optional
        Axis limits ``(min, max)``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 6))

    depths = vel_df["Depth"].values
    vels = vel_df[vel_type].values
    n = len(depths)

    # Step profile
    z_plot, v_plot = [], []
    for i in range(n):
        z_top = depths[i]
        
        if i + 1 < n:
            z_bot = depths[i + 1]
        else:
            # Last layer (half-space) extension
            span = depths[-1] - depths[0]
            if span == 0:
                span = 1000.0  # Fallback for single-layer model
            
            z_def = z_top + span * 0.3
            if ylim:
                # Extend to at least the plot limit if provided
                z_bot = max(z_def, float(max(ylim)))
            else:
                z_bot = z_def

        z_plot.extend([z_top, z_bot])
        v_plot.extend([vels[i], vels[i]])

    ax.plot(v_plot, z_plot, color=color, label=label, **kwargs)
    ax.invert_yaxis()
    ax.set_xlabel(f"{vel_type} (m/s)")
    ax.set_ylabel("Depth (m)")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title("Velocity profile")
    return ax


def rays_2d(
    vel_df: pd.DataFrame,
    rays: Sequence[np.ndarray],
    vel_type: str = "Vp",
    sources: np.ndarray | None = None,
    receivers: np.ndarray | None = None,
    ax=None,
    ray_color: str = "k",
    ray_alpha: float = 0.6,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    plot_model: bool = True,
    add_colorbar: bool = False,
    model_alpha: float = 1.0,
    discrete_colorbar: bool = False,
    **kwargs,
):
    r"""Plot ray paths over a 2-D layered velocity cross-section.

    Parameters
    ----------
    vel_df : pandas.DataFrame
        Velocity model.
    rays : list of numpy.ndarray
        Each element is shape ``(M, 2)`` or ``(M, 3)``.  If 3-D, the
        first two columns are treated as horizontal/depth.
    vel_type : str
        ``'Vp'`` or ``'Vs'`` — used for layer colouring.
    sources, receivers : numpy.ndarray, optional
        Coordinate arrays for plotting markers.
    ax : matplotlib.axes.Axes, optional
    ray_color : str
    ray_alpha : float
    xlim, ylim : tuple, optional
    plot_model : bool
        If *True* (default), plot the velocity model background and
        set axis labels/titles.  If *False*, only plot the rays and
        markers.
    add_colorbar : bool
        If *True* (default *False*), add a colorbar for the velocity
        model. Only applies if *plot_model* is True.
    model_alpha : float
        Opacity of the velocity model layers (0.0 to 1.0). Default 1.0.
    discrete_colorbar : bool
        If *True* (default *False*), quantize the colormap to the
        unique velocity values in the model.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize, BoundaryNorm

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    depths = vel_df["Depth"].values
    vels = vel_df[vel_type].values
    n = len(depths)

    # Determine x-range from rays (only if we need to plot model or set limits)
    if plot_model:
        if rays:
            all_x = np.concatenate([r[:, 0] for r in rays])
            x_lo, x_hi = all_x.min(), all_x.max()
        else:
            x_lo, x_hi = 0, 1000 # Default fallback
            if xlim:
                 x_lo, x_hi = xlim

        pad = (x_hi - x_lo) * 0.05
        x_lo -= pad
        x_hi += pad

        # Layer rectangles
        unique_vels = np.sort(np.unique(vels))
        vmin, vmax = unique_vels[0], unique_vels[-1]
        
        if discrete_colorbar and len(unique_vels) > 1:
            # Create discrete boundaries
            # Midpoints between values
            mids = (unique_vels[:-1] + unique_vels[1:]) / 2.0
            # Extend to cover first and last
            # We can pick abitrary padding, e.g. estimated step
            step = (vmax - vmin) / (len(unique_vels) - 1) if len(unique_vels) > 1 else 1.0
            bounds = np.concatenate(([vmin - step/2], mids, [vmax + step/2]))
            
            cmap = cm.get_cmap("viridis", len(unique_vels))
            norm = BoundaryNorm(bounds, cmap.N)
        else:
            cmap = cm.get_cmap("viridis")
            norm = Normalize(vmin=vmin, vmax=vmax)

        patches = []
        colors_list = []
        for i in range(n):
            z_top = depths[i]
            
            if i + 1 < n:
                z_bot = depths[i + 1]
            else:
                # Last layer (half-space) extension
                span = depths[-1] - depths[0]
                if span == 0: span = 1000.0
                
                z_def = z_top + span * 0.3
                if ylim:
                    z_bot = max(z_def, float(max(ylim)))
                else:
                    z_bot = z_def
            
            rect = Rectangle((x_lo, z_top), x_hi - x_lo, z_bot - z_top)
            patches.append(rect)
            # Use the norm to map velocity to color
            colors_list.append(cmap(norm(vels[i])))

        pc = PatchCollection(patches, facecolor=colors_list, alpha=model_alpha, edgecolor="grey", linewidth=0.5)
        ax.add_collection(pc)
        
        if add_colorbar:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Format ticks for discrete case
            if discrete_colorbar and len(unique_vels) > 1:
                # Place ticks at the unique values
                ticks = unique_vels
                plt.colorbar(sm, ax=ax, label=f"{vel_type} (m/s)", alpha=model_alpha, ticks=ticks)
            else:
                plt.colorbar(sm, ax=ax, label=f"{vel_type} (m/s)", alpha=model_alpha)

    # Rays
    for ray in rays:
        x = ray[:, 0]
        z = ray[:, -1] if ray.shape[1] == 2 else ray[:, 2]
        ax.plot(x, z, color=ray_color, alpha=ray_alpha, linewidth=0.8, **kwargs)

    # Markers
    if sources is not None:
        src = np.atleast_2d(sources)
        ax.scatter(src[:, 0], src[:, -1], marker="*", s=120, c="red", zorder=5, label="Source")
    if receivers is not None:
        rcv = np.atleast_2d(receivers)
        ax.scatter(rcv[:, 0], rcv[:, -1], marker="v", s=60, c="blue", zorder=5, label="Receiver")

    if plot_model:
        ax.invert_yaxis()
        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Ray paths")
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
        
    # Only label legend if we haven't done it manually or if requested
    # But usually the user calls legend() outside.
    # We'll leave the return as is.
    return ax


def rays_3d(
    vel_df: pd.DataFrame,
    rays: Sequence[np.ndarray],
    vel_type: str = "Vp",
    sources: np.ndarray | None = None,
    receivers: np.ndarray | None = None,
    ray_color: str = "red",
    opacity: float = 0.3,
    **kwargs,
):
    r"""Interactive 3-D ray visualisation using Plotly.

    Parameters
    ----------
    vel_df : pandas.DataFrame
        Velocity model.
    rays : list of numpy.ndarray
        Each element is shape ``(M, 3)``.
    vel_type : str
        ``'Vp'`` or ``'Vs'``.
    sources, receivers : numpy.ndarray, optional
        Coordinate arrays for plotting markers.
    ray_color : str
        Ray trace colour.
    opacity : float
        Layer surface opacity.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Rays
    for ray in rays:
        fig.add_trace(
            go.Scatter3d(
                x=ray[:, 0],
                y=ray[:, 1],
                z=ray[:, 2],
                mode="lines",
                line=dict(color=ray_color, width=2),
                showlegend=False,
            )
        )

    # Source / receiver markers
    if sources is not None:
        src = np.atleast_2d(sources)
        fig.add_trace(
            go.Scatter3d(
                x=src[:, 0], y=src[:, 1], z=src[:, 2],
                mode="markers",
                marker=dict(size=6, color="red", symbol="diamond"),
                name="Sources",
            )
        )
    if receivers is not None:
        rcv = np.atleast_2d(receivers)
        fig.add_trace(
            go.Scatter3d(
                x=rcv[:, 0], y=rcv[:, 1], z=rcv[:, 2],
                mode="markers",
                marker=dict(size=4, color="blue", symbol="circle"),
                name="Receivers",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Depth (m)",
            zaxis=dict(autorange="reversed"),
            aspectmode="data",
        ),
        title="3-D Ray paths",
    )
    return fig
