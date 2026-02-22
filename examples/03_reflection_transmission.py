r"""
03. Reflection & transmission
=============================

Reproduction of the classic P-SV reflection & transmission test case from
`Charles J. Ammon's MATLAB Exercise L3 (PDF) <http://eqseis.geosc.psu.edu/cammon/HTML/UsingMATLAB/PDF/ML3%20ReflTransmission.pdf>`_ (:cite:t:`LayWallace1995`, Figure 3.28).

For an incident P-wave the system unknowns are
:math:`[R_{PP},\; R_{PS},\; T_{PP},\; T_{PS}]`.
For an incident SV-wave the unknowns are
:math:`[R_{SP},\; R_{SS},\; T_{SP},\; T_{SS}]`.
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
# Model
# ^^^^^
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
lt.plot.velocity_profile(model_psv, param="Vp", ax=axes[0], ylim=(4000, 0))
lt.plot.velocity_profile(model_psv, param="Vs", ax=axes[1], color="tab:orange", ylim=(4000, 0))
lt.plot.velocity_profile(model_psv, param="Rho", ax=axes[2], color="tab:green", ylim=(4000, 0))

fig.suptitle("P-SV Test Model", fontsize=14)
fig.tight_layout()
plt.show()

# Ray-parameter sweep: p from 0 to 1/Vp_incident
n_p = 200
p_vec = np.linspace(0, 1.0 / mi_vp, n_p + 1)

# Compute all 8 R/T coefficients
RT = lt.psv_rt_coefficients(
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
brew_P = lt.find_brewster_angles(RT, angle_P, keys=["Rpp", "Rps", "Tpp", "Tps"])

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
# We visualize the ray paths for typical situations using `lt.plot.rays_2d`.
# The interface is at 2000 m.


def plot_ray_situation(angle, wave_type, title, ax):
    # 1. Setup background (velocity model) and axes
    # We pass empty rays list first just to set up the plot environment
    lt.plot.rays_2d(
        model_psv, rays=[], ax=ax, vel_type="Vp", 
        xlim=(-100, 6000), ylim=(4000, 0),
        plot_model=True,
        add_colorbar=True,
        model_alpha=0.5,
        discrete_colorbar=True,
    )
    
    # 2. Compute Offset for the given angle to define receiver position
    # The example wants to visualize SPECIFIC angles.
    # trace_rays solves the Two-Point problem (Fixed Receiver).
    # To plot a ray for a specific angle, we first find where it lands.
    # Or we can keep using manual shooting logic?
    # No, the goal is to demonstrate the NEW engine.
    
    # We calculate geometric offset for the flat layers given angle
    v_inc = mi_vp * 1000 if wave_type == "P" else mi_vs * 1000
    p_target = np.sin(np.deg2rad(angle)) / v_inc
    
    # Check critical angles before tracing
    # If p > 1/V_layer, it's evanescent.
    # LayTracer solver handles non-evanescent rays.
    # We manually check evanescence for the legs we want to plot.
    
    source = np.array([0.0, 0.0, 0.0])
    z_int = 2000.0
    z_bot = 4000.0
    
    # Helper to trace and plot one ray variant
    def run_trace(rcv_z, reflection_arg=None, refraction_arg=None, label="", color="", style=""):
        # Calculate theoretical horizontal offset for this p
        # We assume simplified straight rays for this calc (constant layer blocks)
        
        # We need the path legs to calculate X(p_target).
        # We can use lt.offset() if we build the stack manually, 
        # OR just simple trig since model is constant layers.
        
        # Legs depend on reflection/refraction.
        dx = 0.0
        
        # LEG 1: 0 -> 2000
        # Check P-wave layer 0
        v0 = mi_vp * 1000 if wave_type == "P" else mi_vs * 1000
        if p_target * v0 >= 1.0: return # Evanescent at start
        dx += 2000.0 * p_target * v0 / np.sqrt(1.0 - (p_target*v0)**2)
        
        is_refl = (reflection_arg is not None)
        
        if is_refl:
            # LEG 2: 2000 -> 0 (Up)
            # Phase determined by reflection arg "P" or "S"
            ph_up = reflection_arg[0][1]
            v1 = mi_vp * 1000 if ph_up == "P" else mi_vs * 1000
            if p_target * v1 >= 1.0: return # Evanescent reflection
            dx += 2000.0 * p_target * v1 / np.sqrt(1.0 - (p_target*v1)**2)
            z_end = 0.0
        else:
            # LEG 2: 2000 -> 4000 (Down)
            # Phase determined by refraction arg "P" or "S" (or default P/S if None?)
            # trace_rays defaults transmission to same phase if not specified.
            # But here we want to test conversions explicitly.
            # If refraction_arg is set, use it.
            ph_down = refraction_arg[0][1] if refraction_arg else wave_type
            v1 = mt_vp * 1000 if ph_down == "P" else mt_vs * 1000
            
            # Check critical angle for transmission
            if p_target * v1 >= 1.0: return # Critical/Evanescent
            
            dx += (z_bot - z_int) * p_target * v1 / np.sqrt(1.0 - (p_target*v1)**2)
            z_end = z_bot
            
        receiver = np.array([dx, 0.0, z_end])
        
        # RUN THE SOLVER
        try:
            res = lt.trace_rays(
                sources=source,
                receivers=receiver,
                velocity_df=model_psv,
                source_phase=wave_type,
                reflection=reflection_arg,
                refraction=refraction_arg,
                compute_amplitude=False
            )
            
            if res.rays and len(res.rays) > 0 and res.rays[0] is not None:
                lt.plot.rays_2d(
                    model_psv,
                    rays=res.rays,
                    ax=ax,
                    ray_color=color,
                    plot_model=False,
                    linestyle=style,
                    label=label,
                    xlim=(-100, 6000), ylim=(4000, 0)
                )
        except Exception:
            pass # Solver might fail if we messed up bounds, ignore for plot

    # 1. Reflected P
    run_trace(0.0, reflection_arg=[(2000.0, "P")], label="Refl P", color="r", style="--")
    
    # 2. Reflected S
    run_trace(0.0, reflection_arg=[(2000.0, "S")], label="Refl S", color="tab:orange", style=":")
    
    # 3. Transmitted P
    # Note: refraction arg is only needed if MODE CONSTANT changes.
    # P->P is default transmission.
    # But to be explicit we can convert.
    if wave_type == "P":
        run_trace(4000.0, refraction_arg=None, label="Trans P", color="b", style="-")
        run_trace(4000.0, refraction_arg=[(2000.0, "S")], label="Trans S", color="tab:green", style="-.")
    else:
        # Incident S
        run_trace(4000.0, refraction_arg=[(2000.0, "P")], label="Trans P", color="b", style="-")
        run_trace(4000.0, refraction_arg=None, label="Trans S", color="tab:green", style="-.") # S->S

    # Incident ray is not plotted separately because trace_rays returns the FULL path.
    # The previous manual code overlaid legs.
    # The new code plots full V-shapes.
    # This might look slightly different (lines overlapping on the incident leg).
    # That is acceptable and actually more physically correct (showing the full ray).
    
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

RT_sv = lt.psv_rt_coefficients(
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
brew_SV = lt.find_brewster_angles(
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
# .. only:: html
#
#    References
#    ----------
#
#    .. bibliography::
#       :style: unsrt
#       :filter: docname in docnames
#
# .. raw:: html
#
#    <br><br>