r"""
02. Amplitude analysis
======================

This example demonstrates the computation of amplitude-related
quantities alongside ray tracing: the attenuation operator :math:`t^*`,
geometrical spreading, and transmission coefficients.
"""

###############################################################################
# Setup
# -----

import laytracer
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
# `Charles J. Ammon's MATLAB Exercise L3 (PDF) <http://eqseis.geosc.psu.edu/cammon/HTML/UsingMATLAB/PDF/ML3%20ReflTransmission.pdf>`_ (:cite:t:`LayWallace1995`, Figure 3.28).
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

# Medium parameters (Km/s and g/cm^3)
mi_vp, mi_vs, mi_rho = 4.98, 2.9, 2.667   # incident
mt_vp, mt_vs, mt_rho = 8.00, 4.6, 3.38    # transmitted

# Create a DataFrame for visualization (using SI units m/s, kg/m^3)
model_psv = pd.DataFrame({
    "Depth": [0.0, 2000.0],  # Arbitrary interface depth at 2km
    "Vp":    [mi_vp * 1000, mt_vp * 1000],
    "Vs":    [mi_vs * 1000, mt_vs * 1000],
    "Rho":   [mi_rho * 1000, mt_rho * 1000],
})

# Plot the velocity model
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
laytracer.plot.velocity_profile(model_psv, vel_type="Vp", ax=axes[0], ylim=(4000, 0))
laytracer.plot.velocity_profile(model_psv, vel_type="Vs", ax=axes[1], color="tab:orange")
axes[1].set_title("Vs profile")

# Density
z_plot = [0, 2000, 2000, 4000]
r_plot = [mi_rho * 1000, mi_rho * 1000, mt_rho * 1000, mt_rho * 1000]
axes[2].plot(r_plot, z_plot, color="tab:green")
axes[2].invert_yaxis()
axes[2].set_xlabel(r"$\rho$ (kg/m³)")
axes[2].set_title("Density profile")

fig.suptitle("P-SV Test Model", fontsize=14)
fig.tight_layout()
plt.show()

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
#
# For an incident P-wave the ray parameter sweeps from 0 to
# :math:`1/V_P` (grazing P incidence), covering the full 0–90° range.
#
# **Critical angle** (dashed red line):
#
# * Transmitted P becomes evanescent at
#   :math:`\theta_c^{T(P)} = \arcsin(V_P^{(1)}/V_P^{(2)}) \approx 38.5°`.
#   Beyond this angle :math:`|R_{PP}| \to 1` (total reflection).
#   There is no transmitted-SV critical angle because
#   :math:`V_P^{(1)} > V_S^{(2)}` for this model.
#
# **Brewster angles** (dotted purple lines):
#
# * :math:`|R_{PS}|` has a near-zero at ≈37.9°, just before the
#   critical angle.  This is the P-to-SV mode-conversion null,
#   analogous to the optical Brewster angle.  Its position depends
#   on all six elastic parameters, not just the velocity ratio.

# Incidence angle (P-wave): θ = arcsin(p · Vp)
angle_P = np.rad2deg(np.arcsin(np.clip(p_vec * mi_vp, -1, 1)))
crit_P = np.rad2deg(np.arcsin(mi_vp / mt_vp))   # transmitted P critical

# Detect Brewster angles for all P-incident coefficients
brew_P = laytracer.find_brewster_angles(RT, angle_P, keys=["Rpp", "Rps", "Tpp", "Tps"])

# Shared y-limit across all four P-incident panels
p_keys = ["Rpp", "Rps", "Tpp", "Tps"]
ymax_P = max(np.nanmax(np.abs(RT[k])) for k in p_keys) * 1.1
ymax_P = max(ymax_P, 0.5)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "Incident P-wave\n"
    f"Inc: Vp={mi_vp}, Vs={mi_vs}, ρ={mi_rho}  →  "
    f"Trans: Vp={mt_vp}, Vs={mt_vs}, ρ={mt_rho}",
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
               label=f"T(P) crit. {crit_P:.1f}°")
    # Brewster lines for this coefficient
    for ba in brew_P.get(key, []):
        ax.axvline(ba, color="tab:purple", ls=":", lw=0.8,
                   label=f"Brewster {ba:.1f}°")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.05, ymax_P)
    ax.set_xlabel("Incidence angle (°)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
# Ray diagrams (P-incidence)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We visualize the ray paths for typical situations using `laytracer.plot.rays_2d`.
# The interface is at 2000 m.


def make_ray_paths(inc_angle_deg, wave_type="P"):
    """Generate ray paths using LayTracer's engine."""
    # 1. Determine Ray Parameter p
    #    p = sin(theta) / v_incident
    v_inc = mi_vp * 1000 if wave_type == "P" else mi_vs * 1000
    p = np.sin(np.deg2rad(inc_angle_deg)) / v_inc

    # Define interface and bottom depths
    z_int = 2000.0
    z_max = 4000.0
    
    rays_list = []
    labels = []
    colors = []
    styles = []
    
    # Helper to trace a leg
    def trace_leg(p_val, z_start, z_end, v_type, label, color, style, x_start=0.0):
        # Check evanescence: p * v < 1
        # Get velocity of the relevant layer to check critical angle
        # Note: model_psv has depths 0, 2000.
        # Layer 0 (0-2000): incident medium. Layer 1 (2000+): transmitted.
        
        # We need to know which layer we are in to check velocity
        # But for tracing, we can just use laytracer ecosystem
        
        # Build a temporary stack for this segment to calculate offset
        # This implicitly handles the velocity check if we implement it right,
        # or we check manually.
        
        stack = laytracer.build_layer_stack(model_psv, z_start, z_end)
        v_layer = stack.v(v_type)
        if np.any(p_val * v_layer >= 1.0 - 1e-9):
            return None, None # Evanescent

        # Calculate horizontal offset using laytracer's physics
        # 1. q_from_p
        vmax = np.max(v_layer)
        q = laytracer.q_from_p(p_val, vmax)
        
        # 2. offset(q)
        lmd = v_layer / vmax
        dx = laytracer.offset(q, stack.h, lmd)
        
        # 3. Trace ray to exact receiver location
        x_end = x_start + dx
        
        # Create source/receiver coords
        src_pt = np.array([x_start, 0.0, z_start])
        rcv_pt = np.array([x_end, 0.0, z_end])
        
        res = laytracer.trace_rays(
            sources=src_pt,
            receivers=rcv_pt,
            velocity_df=model_psv,
            vel_type=v_type,
            compute_amplitude=False
        )
        
        # Extract ray path
        if res.rays and len(res.rays) > 0:
            return res.rays[0], x_end
        return None, None

    # --- 1. Incident Ray (Surface to Interface) ---
    # Downward in top layer
    ray_inc, x_int = trace_leg(
        p, 0.0, z_int, 
        wave_type, 
        "Incident", "k", "-", x_start=0.0
    )
    
    if ray_inc is None:
        return [], [], [], [] # Should not happen for incident
        
    rays_list.append(ray_inc)
    labels.append("Incident")
    colors.append("k")
    styles.append("-")
    
    # --- 2. Reflected P (Interface to Surface) ---
    # Upward in top layer
    ray_rp, _ = trace_leg(
        p, z_int, 0.0, 
        "Vp", 
        "Refl P", "r", "--", x_start=x_int
    )
    if ray_rp is not None:
        rays_list.append(ray_rp)
        labels.append("Refl P")
        colors.append("r")
        styles.append("--")

    # --- 3. Reflected S (Interface to Surface) ---
    # Upward in top layer
    ray_rs, _ = trace_leg(
        p, z_int, 0.0, 
        "Vs", 
        "Refl S", "tab:orange", ":", x_start=x_int
    )
    if ray_rs is not None:
        rays_list.append(ray_rs)
        labels.append("Refl S")
        colors.append("tab:orange")
        styles.append(":")

    # --- 4. Transmitted P (Interface to Bottom) ---
    # Downward in bottom layer
    ray_tp, _ = trace_leg(
        p, z_int, z_max, 
        "Vp", 
        "Trans P", "b", "-", x_start=x_int
    )
    if ray_tp is not None:
        rays_list.append(ray_tp)
        labels.append("Trans P")
        colors.append("b")
        styles.append("-")

    # --- 5. Transmitted S (Interface to Bottom) ---
    # Downward in bottom layer
    ray_ts, _ = trace_leg(
        p, z_int, z_max, 
        "Vs", 
        "Trans S", "tab:green", "-.", x_start=x_int
    )
    if ray_ts is not None:
        rays_list.append(ray_ts)
        labels.append("Trans S")
        colors.append("tab:green")
        styles.append("-.")
        
    return rays_list, labels, colors, styles



def plot_ray_situation(angle, wave_type, title, ax):
    rays, labels, colors, styles = make_ray_paths(angle, wave_type)
    
    # 1. Setup background (velocity model) and axes
    # We pass a dummy ray to ensure the model plotting logic has a valid range if needed,
    # though with explicit xlim it handles it.
    laytracer.plot.rays_2d(
        model_psv, rays=[], ax=ax, vel_type="Vp", 
        xlim=(-100, 6000), ylim=(4000, 0),
        plot_model=True,
        add_colorbar=True,
        model_alpha=0.5,
        discrete_colorbar=True,
    )
    
    # 2. Plot each ray leg using rays_2d (no background)
    for ray, label, color, style in zip(rays, labels, colors, styles):
        # We need to reshape ray to (M, 3) or (M, 2) list of rays
        # make_ray_paths returns (M, 3) arrays (x, 0, z)
        # rays_2d expects list of arrays.
        laytracer.plot.rays_2d(
            model_psv,
            rays=[ray],
            ax=ax,
            ray_color=color, # This sets the color
            plot_model=False,
            linestyle=style,
            label=label,
            ray_alpha=1.0,
            xlim=(-100, 6000), ylim=(4000, 0) # Maintain limits
        )
    
    ax.legend(loc="lower left", fontsize="small")
    ax.set_title(f"{title}\n(Angle {angle}°)")


# P-incidence scenarios
scenarios_p = [
    (30, "Pre-critical"),
    (45, "Post-critical (Trans P evanescent)"),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for i, (ang, name) in enumerate(scenarios_p):
    plot_ray_situation(ang, "P", name, axes[i])

fig.suptitle("Ray paths: Incident P-wave", fontsize=14)
fig.tight_layout()
plt.show()

#%%

###############################################################################
# Incident SV-wave coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For an incident SV-wave the ray parameter sweeps from 0 to
# :math:`1/V_S` (grazing SV incidence), covering the full 0–90° range.
#
# **Critical angles** (coloured lines) – three distinct thresholds:
#
# * :math:`\theta_c^{T(P)} = \arcsin(V_S^{(1)}/V_P^{(2)}) \approx 21.3°`
#   – transmitted P goes evanescent (blue dotted)
# * :math:`\theta_c^{R(P)} = \arcsin(V_S^{(1)}/V_P^{(1)}) \approx 35.6°`
#   – reflected P goes evanescent (red dashed)
# * :math:`\theta_c^{T(SV)} = \arcsin(V_S^{(1)}/V_S^{(2)}) \approx 39.1°`
#   – transmitted SV goes evanescent (green dash-dot);
#   beyond this angle all energy is reflected as SV
#   (:math:`|R_{SS}| = 1`).
#
# The reflected SV wave is always real (same medium, same velocity).
#
# **Brewster angles** (purple dotted lines) – the near-zeros of
# :math:`|R_{SP}|` near 21° and 40°, and of :math:`|R_{SS}|` near
# 20°, are mode-conversion nulls governed by the full elastic
# contrast.

p_vec_sv = np.linspace(0, 1.0 / mi_vs, n_p + 1)

RT_sv = laytracer.psv_rt_coefficients(
    p=p_vec_sv,
    vp1=mi_vp, vs1=mi_vs, rho1=mi_rho,
    vp2=mt_vp, vs2=mt_vs, rho2=mt_rho,
)

# Incidence angle (SV-wave): θ = arcsin(p · Vs)
angle_SV = np.rad2deg(np.arcsin(np.clip(p_vec_sv * mi_vs, -1, 1)))

# Critical angles
crit_tp = np.rad2deg(np.arcsin(mi_vs / mt_vp))   # transmitted P
crit_rp = np.rad2deg(np.arcsin(mi_vs / mi_vp))   # reflected P
crit_ts = np.rad2deg(np.arcsin(mi_vs / mt_vs))   # transmitted SV

# Detect Brewster angles for all SV-incident coefficients
brew_SV = laytracer.find_brewster_angles(
    RT_sv, angle_SV, keys=["Rsp", "Rss", "Tsp", "Tss"],
)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "Incident SV-wave\n"
    f"Inc: Vp={mi_vp}, Vs={mi_vs}, ρ={mi_rho}  →  "
    f"Trans: Vp={mt_vp}, Vs={mt_vs}, ρ={mt_rho}",
    fontsize=11,
)

labels_sv = [
    (0, 0, "Rsp", r"$|R_{SP}|$",  "Reflected P"),
    (0, 1, "Rss", r"$|R_{SS}|$",  "Reflected SV"),
    (1, 0, "Tsp", r"$|T_{SP}|$",  "Transmitted P"),
    (1, 1, "Tss", r"$|T_{SS}|$",  "Transmitted SV"),
]

# Shared y-limit across all four SV-incident panels
sv_keys = ["Rsp", "Rss", "Tsp", "Tss"]
ymax_SV = max(np.nanmax(np.abs(RT_sv[k])) for k in sv_keys) * 1.1
ymax_SV = max(ymax_SV, 0.5)

for row, col, key, ylabel, title in labels_sv:
    ax = axes[row, col]
    ax.plot(angle_SV, np.abs(RT_sv[key]), "k-", lw=1.5)
    ax.axvline(crit_tp, color="tab:blue", ls=":", lw=0.8,
               label=f"T(P) crit. {crit_tp:.1f}°")
    ax.axvline(crit_rp, color="r", ls="--", lw=0.8,
               label=f"R(P) crit. {crit_rp:.1f}°")
    ax.axvline(crit_ts, color="tab:green", ls="-.", lw=0.8,
               label=f"T(SV) crit. {crit_ts:.1f}°")
    # Brewster lines for this coefficient
    for ba in brew_SV.get(key, []):
        ax.axvline(ba, color="tab:purple", ls=":", lw=0.8,
                   label=f"Brewster {ba:.1f}°")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.05, ymax_SV)
    ax.set_xlabel("Incidence angle (°)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
# Ray diagrams (SV-incidence)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# SV-incidence scenarios
scenarios_sv = [
    (15, "Pre-critical"),
    (25, "Trans P evanescent"),
    (37, "Refl P evanescent"),
    (45, "Trans SV evanescent (Total Reflection)"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
axes = axes.flatten()

for i, (ang, name) in enumerate(scenarios_sv):
    plot_ray_situation(ang, "S", name, axes[i])

fig.suptitle("Ray paths: Incident SV-wave", fontsize=14)
fig.tight_layout()
plt.show()

###############################################################################
# References
# ----------
#
# .. bibliography::
#     :style: unsrt
#     :filter: docname in docnames
#
# .. raw:: html
#
#    <br><br>