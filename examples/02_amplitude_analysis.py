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

print(vel_df)

###############################################################################
# Plot velocity model
# -------------------
#
# Visualise the P-wave velocity, S-wave velocity, and density profiles
# that will be used for the amplitude calculations.

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

laytracer.plot.velocity_profile(vel_df, vel_type="Vp", ax=axes[0])
laytracer.plot.velocity_profile(vel_df, vel_type="Vs", ax=axes[1], color="tab:orange")
axes[1].set_title("Vs profile")

# Density profile (reuse the step-profile pattern)
depths = vel_df["Depth"].values
rho = vel_df["Rho"].values
n = len(depths)
z_plot, r_plot = [], []
for i in range(n):
    z_top = depths[i]
    z_bot = depths[i + 1] if i + 1 < n else z_top + (depths[-1] - depths[0]) * 0.3
    z_plot.extend([z_top, z_bot])
    r_plot.extend([rho[i], rho[i]])

axes[2].plot(r_plot, z_plot, color="tab:green")
axes[2].invert_yaxis()
axes[2].set_xlabel(r"$\rho$ (kg/m³)")
axes[2].set_title("Density profile")

fig.suptitle("Velocity model", fontsize=14)
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
# -----------------------------------
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

#%%

###############################################################################
# P-SV reflection & transmission test
# -----------------------------------
#
# Reproduction of the classic P-SV reflection & transmission test case from
# Charles J. Ammon's MATLAB Exercise L3 (Lay & Wallace, 1995, Figure 3.28).
#
# For an incident P-wave the system unknowns are
# :math:`[R_{PP},\; R_{PS},\; T_{PP},\; T_{PS}]`.
# For an incident SV-wave the unknowns are
# :math:`[R_{SP},\; R_{SS},\; T_{SP},\; T_{SS}]`.
#
# Model
# ^^^^^
# * Incident medium:     Vp = 4.98 km/s,  Vs = 2.9 km/s,  ρ = 2.667 g/cm³
# * Transmitted medium:  Vp = 8.00 km/s,  Vs = 4.6 km/s,  ρ = 3.380 g/cm³

# Medium parameters (units: km/s and g/cm³ — only ratios matter)
mi_vp, mi_vs, mi_rho = 4.98, 2.9, 2.667   # incident
mt_vp, mt_vs, mt_rho = 8.00, 4.6, 3.38    # transmitted

# Ray-parameter sweep: p from 0 to 1/Vp_incident
n_p = 200
p_vec = np.linspace(0, 1.0 / mi_vp, n_p + 1)

# Compute all 8 R/T coefficients
RT = laytracer.psv_rt_coefficients(
    p=p_vec,
    vp1=mi_vp, vs1=mi_vs, rho1=mi_rho,
    vp2=mt_vp, vs2=mt_vs, rho2=mt_rho,
)

###############################################################################
# Incident P-wave coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Incidence angle (P-wave): θ = arcsin(p · Vp)
angle_P = np.rad2deg(np.arcsin(np.clip(p_vec * mi_vp, -1, 1)))
crit_P = np.rad2deg(np.arcsin(mi_vp / mt_vp))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "Incident P-wave\n"
    f"Inc: Vp={mi_vp}, Vs={mi_vs}, ρ={mi_rho}  →  "
    f"Trans: Vp={mt_vp}, Vs={mt_vs}, ρ={mt_rho}  "
    f"(P crit. {crit_P:.1f}°)",
    fontsize=11,
)

labels = [
    (0, 0, "Rpp", r"$|R_{PP}|$",  "Reflected P"),
    (0, 1, "Rps", r"$|R_{PS}|$",  "Reflected SV"),
    (1, 0, "Tpp", r"$|T_{PP}|$",  "Transmitted P"),
    (1, 1, "Tps", r"$|T_{PS}|$",  "Transmitted SV"),
]

for row, col, key, ylabel, title in labels:
    ax = axes[row, col]
    ax.plot(angle_P, np.abs(RT[key]), "k-", lw=1.5)
    ax.axvline(crit_P, color="r", ls="--", lw=0.8,
               label=f"P crit. {crit_P:.1f}°")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.1, 2.0)
    ax.set_xlabel("Incidence angle (°)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

#%%

###############################################################################
# Incident SV-wave coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Incidence angle (SV-wave): θ = arcsin(p · Vs)
angle_SV = np.rad2deg(np.arcsin(np.clip(p_vec * mi_vs, -1, 1)))
crit_SV = np.rad2deg(np.arcsin(mi_vs / mi_vp))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "Incident SV-wave\n"
    f"Inc: Vp={mi_vp}, Vs={mi_vs}, ρ={mi_rho}  →  "
    f"Trans: Vp={mt_vp}, Vs={mt_vs}, ρ={mt_rho}  "
    f"(SV crit. {crit_SV:.1f}°)",
    fontsize=11,
)

labels_sv = [
    (0, 0, "Rsp", r"$|R_{SP}|$",  "Reflected P"),
    (0, 1, "Rss", r"$|R_{SS}|$",  "Reflected SV"),
    (1, 0, "Tsp", r"$|T_{SP}|$",  "Transmitted P"),
    (1, 1, "Tss", r"$|T_{SS}|$",  "Transmitted SV"),
]

for row, col, key, ylabel, title in labels_sv:
    ax = axes[row, col]
    ax.plot(angle_SV, np.abs(RT[key]), "k-", lw=1.5)
    ax.axvline(crit_SV, color="r", ls="--", lw=0.8,
               label=f"SV crit. {crit_SV:.1f}°")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.1, 2.0)
    ax.set_xlabel("Incidence angle (°)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
