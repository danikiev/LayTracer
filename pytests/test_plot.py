import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import laytracer


def test_coefficient_panels_creates_expected_layout():
    angles = np.linspace(0.0, 90.0, 5)
    panels = [
        {
            "curves": [
                {
                    "y": np.linspace(0.0, 1.0, 5),
                    "label": "normalized",
                    "plot_kwargs": {"color": "tab:blue", "lw": 1.5},
                }
            ],
            "markers": [
                {
                    "angle": 30.0,
                    "label": "critical 30.0 deg",
                    "line_kwargs": {"color": "tab:red", "ls": "--", "lw": 0.8},
                }
            ],
            "title": "Panel A",
            "ylabel": r"$|T|$",
            "legend": True,
        },
        {
            "curves": [
                {
                    "y": np.linspace(1.0, 0.0, 5),
                    "plot_kwargs": {"color": "k", "lw": 1.0},
                }
            ],
            "title": "Panel B",
            "ylabel": r"$|R|$",
        },
        {
            "curves": [
                {
                    "y": np.full(5, 0.5),
                    "label": "reference",
                    "plot_kwargs": {"color": "tab:green", "lw": 1.0},
                }
            ],
            "title": "Panel C",
            "ylabel": r"$|T|$",
            "legend": True,
        },
        {
            "curves": [
                {
                    "y": np.linspace(0.2, 0.8, 5),
                    "label": "curve",
                    "plot_kwargs": {"color": "tab:orange", "lw": 1.0},
                }
            ],
            "markers": [{"angle": None}],
            "title": "Panel D",
            "ylabel": r"$|R|$",
            "legend": True,
        },
    ]

    fig, axes = laytracer.plot.coefficient_panels(
        panels,
        shape=(2, 2),
        default_x=angles,
        default_xlim=(0.0, 90.0),
        default_ylim=(-0.05, 1.05),
        default_xlabel="Incidence angle (deg)",
        suptitle="Coefficient diagnostics",
    )

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == "Panel A"
    assert axes[0, 0].get_ylabel() == r"$|T|$"
    assert axes[1, 1].get_xlabel() == "Incidence angle (deg)"
    assert len(axes[0, 0].lines) == 2
    assert axes[0, 0].get_legend() is not None
    assert fig._suptitle.get_text() == "Coefficient diagnostics"

    plt.close(fig)


def test_coefficient_panels_validates_panel_count():
    with pytest.raises(ValueError):
        laytracer.plot.coefficient_panels([], shape=(1, 1))


def test_coefficient_panels_styles_complex_and_evanescent_segments():
    angles = np.linspace(0.0, 40.0, 5)
    coeff = np.array([
        1.0 + 0.0j,
        0.8 + 0.0j,
        0.7 + 1e-6j,
        0.6 + 1e-6j,
        0.5 + 1e-6j,
    ])
    evanescent = np.array([False, False, False, True, True])
    panels = [
        {
            "curves": [
                {
                    "y": np.abs(coeff),
                    "complex_from": coeff,
                    "evanescent_mask": evanescent,
                    "plot_kwargs": {"color": "k", "lw": 1.0},
                }
            ],
        }
    ]

    fig, axes = laytracer.plot.coefficient_panels(
        panels,
        shape=(1, 1),
        default_x=angles,
    )

    assert len(axes[0, 0].lines) == 3
    assert axes[0, 0].lines[0].get_linestyle() == "-"
    assert axes[0, 0].lines[1].get_linestyle() == "--"
    assert axes[0, 0].lines[2].get_linestyle() == "-."
    np.testing.assert_array_equal(
        np.isnan(axes[0, 0].lines[0].get_ydata()),
        np.array([False, False, True, True, True]),
    )
    np.testing.assert_array_equal(
        np.isnan(axes[0, 0].lines[1].get_ydata()),
        np.array([True, False, False, True, True]),
    )
    np.testing.assert_array_equal(
        np.isnan(axes[0, 0].lines[2].get_ydata()),
        np.array([True, True, False, False, False]),
    )

    plt.close(fig)


def test_coefficient_panels_validates_segment_shape():
    panels = [
        {
            "curves": [
                {
                    "y": np.ones(3),
                    "complex_from": np.ones(2, dtype=complex),
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="same shape"):
        laytracer.plot.coefficient_panels(
            panels,
            shape=(1, 1),
            default_x=np.arange(3),
        )


def test_rays_2d_accepts_custom_layer_colors_and_linewidth():
    vel_df = pd.DataFrame({
        "Depth": [0.0, 1000.0],
        "Vp": [2000.0, 3000.0],
    })
    ray = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    layer_colors = ["#fdc086", "#ffff99"]

    fig, ax = plt.subplots()
    laytracer.plot.rays_2d(
        vel_df,
        rays=[ray],
        ax=ax,
        xlim=(0.0, 1000.0),
        ylim=(1200.0, 0.0),
        discrete_colorbar=True,
        layer_colors=layer_colors,
        ray_linewidth=2.2,
    )

    facecolors = ax.collections[0].get_facecolors()
    expected = np.array([matplotlib.colors.to_rgba(color) for color in layer_colors])
    np.testing.assert_allclose(facecolors[:2], expected)
    assert ax.lines[0].get_linewidth() == 2.2

    plt.close(fig)


def test_rays_2d_uses_ray_extent_without_x_padding():
    vel_df = pd.DataFrame({
        "Depth": [0.0, 1000.0],
        "Vp": [2000.0, 3000.0],
    })
    ray = np.array([[0.0, 0.0], [1000.0, 1000.0]])

    fig, ax = plt.subplots()
    laytracer.plot.rays_2d(
        vel_df,
        rays=[ray],
        ax=ax,
        ylim=(1200.0, 0.0),
    )

    assert ax.get_xlim() == (0.0, 1000.0)

    plt.close(fig)
