r"""
04. Amplitude analysis
======================

This example demonstrates the computation of amplitude-related
quantities alongside ray tracing: the attenuation operator :math:`t^*`,
geometrical spreading, and transmission coefficients.
"""

###############################################################################
# Setup
# -----

import laytracer as lt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 2

###############################################################################
# Define velocity model
# ---------------------

vel_df = pd.DataFrame({
    "Depth": [0.0, 1000.0, 2000.0, 3500.0],
    "Vp":    [3000.0, 4500.0, 5500.0, 6500.0],
    "Vs":    [1500.0, 2250.0, 2750.0, 3250.0],
    "Rho":   [2200.0, 2500.0, 2700.0, 2900.0],
    "Qp":    [200.0,  50.0,   600.0,  800.0],
    "Qs":    [100.0,  25.0,   300.0,  400.0],
})

print(vel_df)

###############################################################################
# Plot velocity model
# -------------------
#
# Visualise the parameters of the model (P-wave, S-wave, density, attenuation)
# that will be used for the amplitude calculations.

fig, axes = plt.subplots(1, 4, figsize=(12, 5), sharey=True)

lt.plot.velocity_profile(vel_df, param="Vp", ax=axes[0])
lt.plot.velocity_profile(vel_df, param="Vs", ax=axes[1], color="tab:orange")
lt.plot.velocity_profile(vel_df, param="Rho", ax=axes[2], color="tab:green")
lt.plot.velocity_profile(vel_df, param="Qp", ax=axes[3], color="tab:purple")

fig.suptitle("Model profiles", fontsize=14)
fig.tight_layout()
plt.show()

#%%

###############################################################################
# Trace rays with amplitude computation
# -------------------------------------
#
# Trace P-waves from a deep source to receivers at varying offsets,
# requesting :math:`t^*`, spreading, and transmission.

src = np.array([0.0, 0.0, 3000.0])

offsets = np.arange(500, 15001, 500)
rcvs = np.column_stack([offsets, np.zeros_like(offsets), np.zeros_like(offsets)])

result = lt.trace_rays(
    sources=src,
    receivers=rcvs,
    velocity_df=vel_df,
    source_phase="P",
    compute_amplitude=True,
    transcoef_method="angle",
)

###############################################################################
# Plot ray paths
# --------------
#
# Before analysing the amplitudes, let's visualize the ray paths from the
# source to the receivers. We overlay the rays on the P-wave velocity model
# to observe their trajectories.

fig, ax = plt.subplots(figsize=(10, 6))
lt.plot.rays_2d(
    vel_df,
    rays=result.rays,
    sources=src,
    receivers=rcvs,
    ax=ax,
    vel_type="Vp",
    plot_model=True,
    add_colorbar=True,
    model_alpha=0.6,
    discrete_colorbar=True,
    unit="km",
)
ax.set_title("Ray paths from deep source to surface receivers")
fig.tight_layout()
plt.show()

#%%

###############################################################################
# Plot amplitude quantities vs offset
# -----------------------------------
#
# Here we analyze the variation of different amplitude-related quantities as a 
# function of receiver offset:
# 
# * **Travel time**: Increases smoothly with offset. The curvature is governed 
#   by the velocity structure (moveout equation).
# * **Attenuation operator** :math:`t^*`: Calculated as the path integral 
#   :math:`t^* = \int_{\mathrm{ray}} \frac{dt}{Q(s)}`. It represents cumulative 
#   anelastic decay. Rays traveling further horizontally spend more time traversing 
#   the highly attenuating layer (Q=50) between 1-2 km depth, accumulating higher 
#   :math:`t^*`.
# * **Geometrical spreading**: Measures the spatial divergence of the energetic 
#   ray tube. It generally grows with propagation distance, but velocity contrasts 
#   distort wavefronts, causing focusing or defocusing effects.
# * **Transmission coefficient product**: The cumulative product of Zoeppritz
#   transmission coefficients :math:`\prod |T_k|` across all crossed interfaces.
#   Notice how the transmission efficiency drops sharply at larger offsets as the 
#   rays become more grazing, converting more energy into reflected modes.

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

result_normal = lt.trace_rays(
    sources=src,
    receivers=rcvs,
    velocity_df=vel_df,
    source_phase="P",
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
