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
# -----
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


def _marker_spec(angle, label, color, linestyle, linewidth=0.8):
    return {
        "angle": angle,
        "label": label,
        "line_kwargs": {
            "color": color,
            "ls": linestyle,
            "lw": linewidth,
        },
    }


def _coefficient_panels(panel_defs, curve_defs, common_markers, panel_markers, ylim):
    panels = []
    for key, ylabel, title in panel_defs:
        curves = []
        for curve_def in curve_defs:
            curves.append(
                {
                    "y": np.abs(curve_def["data"][key]),
                    "label": curve_def.get("label"),
                    "plot_kwargs": dict(curve_def["plot_kwargs"]),
                }
            )

        markers = [dict(marker) for marker in common_markers]
        markers.extend(panel_markers.get(key, []))

        panels.append(
            {
                "curves": curves,
                "markers": markers,
                "title": title,
                "ylabel": ylabel,
                "ylim": ylim,
                "legend": True,
            }
        )

    return panels

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
# ----------------------------
#
# For an incident P-wave the ray parameter sweeps from 0 to
# :math:`1/V_P` (grazing P incidence), covering the full :math:`0--90^{\circ}` range.
#
# **Critical angle** (dashed red line):
#
# * Transmitted P becomes evanescent at
#   :math:`\theta_c^{T(P)} = \arcsin(V_P^{(1)}/V_P^{(2)}) \approx 38.5^{\circ}`.
#   Beyond this angle :math:`|R_{PP}| \to 1` (total reflection).
#   There is no transmitted-SV critical angle because
#   :math:`V_P^{(1)} > V_S^{(2)}` for this model.
#
# **Brewster angles** (dotted purple lines):
#
# * :math:`|R_{PS}|` has a near-zero at :math:`37.9^{\circ}`, just before the
#   critical angle.  This is the P-to-SV mode-conversion null,
#   analogous to the optical Brewster angle.  Its position depends
#   on all six elastic parameters, not just the velocity ratio.

# Incidence angle (P-wave): :math:`\theta` = \arcsin{p \cdot V_p}`
angle_P = np.rad2deg(np.arcsin(np.clip(p_vec * mi_vp, -1, 1)))
crit_P = lt.find_critical_angles(mi_vp, {"T(P)": mt_vp})

# Detect Brewster angles for all P-incident coefficients
brew_P = lt.find_brewster_angles(RT, angle_P, keys=["Rpp", "Rps", "Tpp", "Tps"])

# Shared y-limit across all four P-incident panels
p_keys = ["Rpp", "Rps", "Tpp", "Tps"]
ymax_P = max(np.nanmax(np.abs(RT[k])) for k in p_keys) * 1.1
ymax_P = max(ymax_P, 0.5)

labels = [
    ("Rpp", r"$|R_{PP}|$", "Reflected P"),
    ("Rps", r"$|R_{PS}|$", "Reflected SV"),
    ("Tpp", r"$|T_{PP}|$", "Transmitted P"),
    ("Tps", r"$|T_{PS}|$", "Transmitted SV"),
]
common_markers_P = [
    _marker_spec(
        crit_P["T(P)"],
        f"T(P) crit. {crit_P['T(P)']:.1f} deg",
        "r",
        "--",
    )
]
panel_markers_P = {
    key: [
        _marker_spec(ba, f"Brewster {ba:.1f} deg", "tab:purple", ":")
        for ba in brew_P.get(key, [])
    ]
    for key in p_keys
}
panels = _coefficient_panels(
    labels,
    curve_defs=[{"data": RT, "plot_kwargs": {"color": "k", "lw": 1.5}}],
    common_markers=common_markers_P,
    panel_markers=panel_markers_P,
    ylim=(-0.05, ymax_P),
)
fig, axes = lt.plot.coefficient_panels(
    panels,
    shape=(2, 2),
    figsize=(12, 9),
    default_x=angle_P,
    default_xlim=(0.0, 90.0),
    default_xlabel="Incidence angle (deg)",
    suptitle=(
        "Incident P-wave\n"
        f"Inc: Vp={mi_vp}, Vs={mi_vs}, rho={mi_rho}  ->  "
        f"Trans: Vp={mt_vp}, Vs={mt_vs}, rho={mt_rho}"
    ),
)
plt.show()

###############################################################################
# Normalized P-wave coefficients
# ------------------------------
#
# Energy-flux-normalized coefficients account for the impedance and
# directional cosine contrast across the interface.  They are useful
# for amplitude-preserving modelling because the product of
# normalized transmission coefficients along a ray is the
# displacement-amplitude transfer factor that conserves energy flux.
#
# The normalization follows :cite:t:`Cerveny2001` Eq. 5.3.10:
#
# .. math::
#    R_{mn}^\text{norm} = \bar{R}_{mn}
#    \sqrt{\frac{V_{\text{out}}\,\rho_{\text{out}}\,
#          \cos\theta_{\text{out}}}
#         {V_{\text{in}}\,\rho_{\text{in}}\,
#          \cos\theta_{\text{in}}}}

# Mapping: key -> (v_in, rho_in, v_out, rho_out)
norm_map_P = {
    "Rpp": (mi_vp, mi_rho, mi_vp, mi_rho),
    "Rps": (mi_vp, mi_rho, mi_vs, mi_rho),
    "Tpp": (mi_vp, mi_rho, mt_vp, mt_rho),
    "Tps": (mi_vp, mi_rho, mt_vs, mt_rho),
}

RT_norm_P = {}
for key, (vi, ri, vo, ro) in norm_map_P.items():
    RT_norm_P[key] = lt.normalize_rt_coefficient(
        RT[key], p_vec, vi, ri, vo, ro,
    )

ymax_Pn = max(
    np.nanmax(np.abs(RT_norm_P[k])) for k in p_keys
) * 1.1
ymax_Pn = max(ymax_Pn, 0.5)

panels = _coefficient_panels(
    labels,
    curve_defs=[
        {
            "data": RT,
            "label": "standard",
            "plot_kwargs": {"color": "k", "lw": 0.8, "alpha": 0.4},
        },
        {
            "data": RT_norm_P,
            "label": "normalized",
            "plot_kwargs": {"color": "tab:blue", "lw": 1.5},
        },
    ],
    common_markers=common_markers_P,
    panel_markers=panel_markers_P,
    ylim=(-0.05, max(ymax_P, ymax_Pn)),
)
fig, axes = lt.plot.coefficient_panels(
    panels,
    shape=(2, 2),
    figsize=(12, 9),
    default_x=angle_P,
    default_xlim=(0.0, 90.0),
    default_xlabel="Incidence angle (deg)",
    suptitle=(
        "Incident P-wave - normalized (Cerveny, 2001)\n"
        f"Inc: Vp={mi_vp}, Vs={mi_vs}, rho={mi_rho}  ->  "
        f"Trans: Vp={mt_vp}, Vs={mt_vs}, rho={mt_rho}"
    ),
)
plt.show()

# %%
# Ray diagrams (P-incidence)
# --------------------------
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
                requested={"travel_times", "rays", "ray_parameters"}
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
# -----------------------------
#
# For an incident SV-wave the ray parameter sweeps from 0 to
# :math:`1/V_S` (grazing SV incidence), covering the full :math:`0--90^{\circ}` range.
#
# **Critical angles** (coloured lines) - three distinct thresholds:
#
# * :math:`\theta_c^{T(P)} = \arcsin(V_S^{(1)}/V_P^{(2)}) \approx 21.3^{\circ}`
#   - transmitted P goes evanescent (blue dotted)
# * :math:`\theta_c^{R(P)} = \arcsin(V_S^{(1)}/V_P^{(1)}) \approx 35.6^{\circ}`
#   - reflected P goes evanescent (red dashed)
# * :math:`\theta_c^{T(SV)} = \arcsin(V_S^{(1)}/V_S^{(2)}) \approx 39.1^{\circ}`
#   - transmitted SV goes evanescent (green dash-dot);
#   beyond this angle all energy is reflected as SV
#   (:math:`|R_{SS}| = 1`).
#
# The reflected SV wave is always real (same medium, same velocity).
#
# **Brewster angles** (purple dotted lines) - the near-zeros of
# :math:`|R_{SP}|` near 21° and 40°, and of :math:`|R_{SS}|` near
# 20°, are mode-conversion nulls governed by the full elastic
# contrast.

p_vec_sv = np.linspace(0, 1.0 / mi_vs, n_p + 1)

RT_sv = lt.psv_rt_coefficients(
    p=p_vec_sv,
    vp1=mi_vp, vs1=mi_vs, rho1=mi_rho,
    vp2=mt_vp, vs2=mt_vs, rho2=mt_rho,
)

# Incidence angle (SV-wave):  :math:`\theta = \arcsin(p \cdot V_s)`
angle_SV = np.rad2deg(np.arcsin(np.clip(p_vec_sv * mi_vs, -1, 1)))

# Critical angles
crit_SV = lt.find_critical_angles(
    mi_vs,
    {"T(P)": mt_vp, "R(P)": mi_vp, "T(SV)": mt_vs},
)

# Detect Brewster angles for all SV-incident coefficients
brew_SV = lt.find_brewster_angles(
    RT_sv, angle_SV, keys=["Rsp", "Rss", "Tsp", "Tss"],
)

labels_sv = [
    ("Rsp", r"$|R_{SP}|$", "Reflected P"),
    ("Rss", r"$|R_{SS}|$", "Reflected SV"),
    ("Tsp", r"$|T_{SP}|$", "Transmitted P"),
    ("Tss", r"$|T_{SS}|$", "Transmitted SV"),
]

# Shared y-limit across all four SV-incident panels
sv_keys = ["Rsp", "Rss", "Tsp", "Tss"]
ymax_SV = max(np.nanmax(np.abs(RT_sv[k])) for k in sv_keys) * 1.1
ymax_SV = max(ymax_SV, 0.5)

common_markers_SV = [
    _marker_spec(crit_SV["T(P)"], f"T(P) crit. {crit_SV['T(P)']:.1f} deg", "tab:blue", ":"),
    _marker_spec(crit_SV["R(P)"], f"R(P) crit. {crit_SV['R(P)']:.1f} deg", "r", "--"),
    _marker_spec(crit_SV["T(SV)"], f"T(SV) crit. {crit_SV['T(SV)']:.1f} deg", "tab:green", "-."),
]
panel_markers_SV = {
    key: [
        _marker_spec(ba, f"Brewster {ba:.1f} deg", "tab:purple", ":")
        for ba in brew_SV.get(key, [])
    ]
    for key in sv_keys
}
panels = _coefficient_panels(
    labels_sv,
    curve_defs=[{"data": RT_sv, "plot_kwargs": {"color": "k", "lw": 1.5}}],
    common_markers=common_markers_SV,
    panel_markers=panel_markers_SV,
    ylim=(-0.05, ymax_SV),
)
fig, axes = lt.plot.coefficient_panels(
    panels,
    shape=(2, 2),
    figsize=(12, 9),
    default_x=angle_SV,
    default_xlim=(0.0, 90.0),
    default_xlabel="Incidence angle (deg)",
    suptitle=(
        "Incident SV-wave\n"
        f"Inc: Vp={mi_vp}, Vs={mi_vs}, rho={mi_rho}  ->  "
        f"Trans: Vp={mt_vp}, Vs={mt_vs}, rho={mt_rho}"
    ),
)
plt.show()

###############################################################################
# Normalized SV-wave coefficients (Červený, 2001)
# ------------------------------------------------
#
# Same energy-flux normalization applied to the SV-incident
# coefficients.  The three critical-angle markers are preserved.

# Mapping: key -> (v_in, rho_in, v_out, rho_out)
norm_map_SV = {
    "Rsp": (mi_vs, mi_rho, mi_vp, mi_rho),
    "Rss": (mi_vs, mi_rho, mi_vs, mi_rho),
    "Tsp": (mi_vs, mi_rho, mt_vp, mt_rho),
    "Tss": (mi_vs, mi_rho, mt_vs, mt_rho),
}

RT_norm_SV = {}
for key, (vi, ri, vo, ro) in norm_map_SV.items():
    RT_norm_SV[key] = lt.normalize_rt_coefficient(
        RT_sv[key], p_vec_sv, vi, ri, vo, ro,
    )

ymax_SVn = max(
    np.nanmax(np.abs(RT_norm_SV[k])) for k in sv_keys
) * 1.1
ymax_SVn = max(ymax_SVn, 0.5)

panels = _coefficient_panels(
    labels_sv,
    curve_defs=[
        {
            "data": RT_sv,
            "label": "standard",
            "plot_kwargs": {"color": "k", "lw": 0.8, "alpha": 0.4},
        },
        {
            "data": RT_norm_SV,
            "label": "normalized",
            "plot_kwargs": {"color": "tab:blue", "lw": 1.5},
        },
    ],
    common_markers=common_markers_SV,
    panel_markers=panel_markers_SV,
    ylim=(-0.05, max(ymax_SV, ymax_SVn)),
)
fig, axes = lt.plot.coefficient_panels(
    panels,
    shape=(2, 2),
    figsize=(12, 9),
    default_x=angle_SV,
    default_xlim=(0.0, 90.0),
    default_xlabel="Incidence angle (deg)",
    suptitle=(
        "Incident SV-wave - normalized (Cerveny, 2001)\n"
        f"Inc: Vp={mi_vp}, Vs={mi_vs}, rho={mi_rho}  ->  "
        f"Trans: Vp={mt_vp}, Vs={mt_vs}, rho={mt_rho}"
    ),
)
plt.show()

# %%
# Ray diagrams (SV-incidence)
# ---------------------------

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
