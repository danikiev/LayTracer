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
    param: str = "Vp",
    ax=None,
    color: str | None = None,
    label: str | None = None,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    unit: str = "m",
    **kwargs,
):
    r"""Plot a 1-D model parameter–depth step profile.

    Parameters
    ----------
    vel_df : pandas.DataFrame
        Velocity model.
    param : str, optional
        ``'Vp'`` (default), ``'Vs'``, ``'Rho'``, ``'Qp'``, or ``'Qs'``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  Created if *None*.
    color : str, optional
        Line colour.
    label : str, optional
        Legend label.
    xlim, ylim : tuple, optional
        Axis limits ``(min, max)``.
    unit : str, optional
        ``'m'`` (default) or ``'km'``. Scales the vertical depth axis.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 6))

    depths = vel_df["Depth"].values
    vals = vel_df[param].values
    n = len(depths)
    
    scale = 1000.0 if unit.lower() == "km" else 1.0

    # Step profile
    z_plot, v_plot = [], []
    for i in range(n):
        z_top = depths[i] / scale
        
        if i + 1 < n:
            z_bot = depths[i + 1] / scale
        else:
            # Last layer (half-space) extension
            span = (depths[-1] - depths[0]) / scale
            if span == 0:
                span = 1000.0 / scale  # Fallback for single-layer model
            
            z_def = z_top + span * 0.3
            if ylim:
                # Extend to at least the plot limit if provided
                z_bot = max(z_def, float(max(ylim)))
            else:
                z_bot = z_def

        z_plot.extend([z_top, z_bot])
        v_plot.extend([vals[i], vals[i]])

    ax.plot(v_plot, z_plot, color=color, label=label, **kwargs)
    
    if param in ("Vp", "Vs"):
        xlabel_str = f"{param} (m/s)"
    elif param == "Rho":
        xlabel_str = r"$\rho$ (kg/m³)"
    elif param in ("Qp", "Qs"):
        xlabel_str = f"{param}"
    else:
        xlabel_str = param

    ax.margins(y=0)
    ax.set_xlabel(xlabel_str)
    ax.set_ylabel(f"Depth ({unit})")
    
    # Handle y-axis limits and inversion
    if ylim:
        ax.set_ylim(ylim)
    
    # Only invert if top is less than bottom (matplotlib default puts 0 at bottom)
    bottom, top = ax.get_ylim()
    if bottom < top:
        ax.invert_yaxis()
        
    if xlim:
        ax.set_xlim(xlim)
        
    title_str = "Velocity profile" if param in ("Vp", "Vs") else f"{param} profile"
    ax.set_title(title_str)
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
    unit: str = "m",
    plot_model: bool = True,
    add_colorbar: bool = False,
    discrete_colorbar: bool = False,
    model_alpha: float = 1.0,    
    equal_scale: bool = True,
    colorbar_orientation: str = "vertical",
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
    unit : str
        ``'m'`` (default) or ``'km'``. Scales coordinates and labels.
    add_colorbar : bool
        If *True* (default *False*), add a colorbar for the velocity
        model. Only applies if *plot_model* is True.
    discrete_colorbar : bool
        If *True* (default *False*), quantize the colormap to the
        unique velocity values in the model.
    model_alpha : float
        Opacity of the velocity model layers (0.0 to 1.0). Default 1.0.
    equal_scale : bool
        If *True* (default *True*), force equal scaling for x and y axes
        using ``ax.set_aspect('equal')``.
    colorbar_orientation : str
        ``'vertical'`` (default) or ``'horizontal'``.

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
    
    scale = 1000.0 if unit.lower() == "km" else 1.0

    # Determine x-range from rays (only if we need to plot model or set limits)
    # Determine x-range from rays (only if we need to plot model or set limits)
    if plot_model:
        if rays:
            all_x = np.concatenate([r[:, 0] / scale for r in rays])
            x_lo, x_hi = all_x.min(), all_x.max()
        else:
            x_lo, x_hi = 0, 1000 / scale # Default fallback
            if xlim:
                 x_lo, x_hi = sorted(xlim)

        pad = (x_hi - x_lo) * 0.05
        x_lo -= pad
        x_hi += pad
        
        # Ensure model background covers the requested xlim if provided
        if xlim:
            x_lo = min(x_lo, min(xlim))
            x_hi = max(x_hi, max(xlim))

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
        elif discrete_colorbar and len(unique_vels) == 1:
            # Single unique velocity — one discrete colour, tick at exact value
            bounds = np.array([vmin - 0.5, vmax + 0.5])
            cmap = cm.get_cmap("viridis", 1)
            norm = BoundaryNorm(bounds, cmap.N)
        else:
            cmap = cm.get_cmap("viridis")
            # Guard against vmin == vmax (e.g. single-velocity model)
            if vmin == vmax:
                norm = Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

        # Determine the deepest point across all rays, sources, and receivers
        # so the half-space layer rectangle always covers the visible area.
        z_max_data = 0.0
        if rays:
            all_z = np.concatenate(
                [(r[:, -1] if r.shape[1] == 2 else r[:, 2]) / scale for r in rays]
            )
            z_max_data = max(z_max_data, float(all_z.max()))
        if sources is not None:
            z_max_data = max(z_max_data, float(np.atleast_2d(sources)[:, -1].max() / scale))
        if receivers is not None:
            z_max_data = max(z_max_data, float(np.atleast_2d(receivers)[:, -1].max() / scale))

        patches = []
        colors_list = []
        for i in range(n):
            z_top = depths[i] / scale
            
            
            if i + 1 < n:
                z_bot = depths[i + 1] / scale
            else:
                # Last layer (half-space) extension
                span = (depths[-1] - depths[0]) / scale
                if span == 0: span = 1000.0 / scale
                
                z_def = z_top + span * 0.3
                # Extend to cover the deepest data point (with 10 % padding)
                z_def = max(z_def, z_max_data * 1.1)
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
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            if colorbar_orientation == "horizontal":
                cax = divider.append_axes("bottom", size="5%", pad=0.5)
            else:
                cax = divider.append_axes("right", size="5%", pad=0.1)

            # Format ticks for discrete case
            if discrete_colorbar and len(unique_vels) >= 1:
                # Place ticks at the unique values
                ticks = unique_vels
                plt.colorbar(sm, cax=cax, orientation=colorbar_orientation, label=f"{vel_type} (m/s)", alpha=model_alpha, ticks=ticks)
            else:
                plt.colorbar(sm, cax=cax, orientation=colorbar_orientation, label=f"{vel_type} (m/s)", alpha=model_alpha)

    # Rays
    for ray in rays:
        x = ray[:, 0] / scale
        z = (ray[:, -1] if ray.shape[1] == 2 else ray[:, 2]) / scale
        ax.plot(x, z, color=ray_color, alpha=ray_alpha, linewidth=0.8, **kwargs)

    # Markers
    if sources is not None:
        src = np.atleast_2d(sources)
        ax.scatter(src[:, 0] / scale, src[:, -1] / scale, marker="*", s=120, c="red", zorder=5, label="Source")
    if receivers is not None:
        rcv = np.atleast_2d(receivers)
        ax.scatter(rcv[:, 0] / scale, rcv[:, -1] / scale, marker="v", s=60, c="blue", zorder=5, label="Receiver")

    ax.margins(y=0)

    if plot_model:
        ax.invert_yaxis()
        ax.set_xlabel(f"Horizontal distance ({unit})")
        ax.set_ylabel(f"Depth ({unit})")
        ax.set_title("Ray paths")
    
    if xlim:
        ax.set_xlim(xlim)
    elif plot_model:
        ax.set_xlim(x_lo, x_hi)

    if ylim:
        ax.set_ylim(ylim)
    elif plot_model:
        # Default depth range from model (0 to bottom)
        # Find max depth of model or rays
        z_max_model = depths[-1] / scale
        if rays:
             all_z = np.concatenate([(r[:, -1] if r.shape[1] == 2 else r[:, 2]) / scale for r in rays])
             z_max_rays = all_z.max()
             z_max_model = max(z_max_model, z_max_rays)
        
        # Add slight padding at bottom
        z_max_model *= 1.1
        ax.set_ylim(z_max_model, 0)
        
    if equal_scale:
        ax.set_aspect("equal")

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
