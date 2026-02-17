r"""
Amplitude analysis: t*, spreading, and transmission
=====================================================

This example demonstrates the computation of amplitude-related
quantities alongside ray tracing: the attenuation operator :math:`t^*`,
geometrical spreading, and transmission coefficients.
"""

###############################################################################
# Setup
# -----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import laytracer

###############################################################################
# Define velocity model
# ---------------------

vel_df = pd.DataFrame({
    "Depth": [0.0, 1000.0, 2000.0, 3500.0],
    "Vp":    [3000.0, 4500.0, 5500.0, 6500.0],
    "Vs":    [1500.0, 2250.0, 2750.0, 3250.0],
    "Rho":   [2200.0, 2500.0, 2700.0, 2900.0],
    "Qp":    [200.0,  400.0,  600.0,  800.0],
    "Qs":    [100.0,  200.0,  300.0,  400.0],
})

###############################################################################
# Trace rays with amplitude computation
# --------------------------------------
#
# Trace P-waves from a deep source to receivers at varying offsets,
# requesting :math:`t^*`, spreading, and transmission.

src = np.array([0.0, 0.0, 3000.0])

offsets = np.arange(500, 15001, 500)
rcvs = np.column_stack([offsets, np.zeros_like(offsets), np.zeros_like(offsets)])

result = laytracer.trace_rays(
    sources=src,
    receivers=rcvs,
    velocity_df=vel_df,
    vel_type="Vp",
    compute_amplitude=True,
    transcoef_method="angle",
)

###############################################################################
# Plot amplitude quantities vs offset
# ------------------------------------
#
# We plot travel time, :math:`t^*`, geometrical spreading, and
# transmission coefficient product side by side.

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(offsets / 1000, result.travel_times, "o-", markersize=3)
axes[0, 0].set_xlabel("Offset (km)")
axes[0, 0].set_ylabel("Travel time (s)")
axes[0, 0].set_title("Travel time")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(offsets / 1000, result.tstar, "o-", markersize=3, color="tab:orange")
axes[0, 1].set_xlabel("Offset (km)")
axes[0, 1].set_ylabel(r"$t^*$ (s)")
axes[0, 1].set_title(r"Attenuation operator $t^*$")
axes[0, 1].grid(True, alpha=0.3)

if result.spreading is not None:
    valid = result.spreading > 0
    axes[1, 0].plot(
        offsets[valid] / 1000, result.spreading[valid],
        "o-", markersize=3, color="tab:green",
    )
axes[1, 0].set_xlabel("Offset (km)")
axes[1, 0].set_ylabel("Spreading factor")
axes[1, 0].set_title("Geometrical spreading")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(
    offsets / 1000, result.trans_product,
    "o-", markersize=3, color="tab:red",
)
axes[1, 1].set_xlabel("Offset (km)")
axes[1, 1].set_ylabel(r"$\prod |T_k|$")
axes[1, 1].set_title("Transmission coefficient product")
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle("Amplitude quantities vs. offset", fontsize=14)
fig.tight_layout()
plt.show()

#%%

###############################################################################
# Compare normal-incidence vs angle-dependent transmission
# --------------------------------------------------------

result_normal = laytracer.trace_rays(
    sources=src,
    receivers=rcvs,
    velocity_df=vel_df,
    vel_type="Vp",
    compute_amplitude=True,
    transcoef_method="normal",
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(offsets / 1000, result.trans_product, "o-", label="Zoeppritz", markersize=3)
ax.plot(offsets / 1000, result_normal.trans_product, "s-", label="Normal incidence", markersize=3)
ax.set_xlabel("Offset (km)")
ax.set_ylabel(r"$\prod |T_k|$")
ax.set_title("Transmission: Zoeppritz vs. normal incidence")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
