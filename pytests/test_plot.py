import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
