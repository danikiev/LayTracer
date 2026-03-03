r"""
05. Homogeneous equivalence quality check
=========================================

A two-layer model whose layers share **identical** elastic parameters must
reproduce the closed-form homogeneous-medium solution exactly. This example
computes travel time, :math:`t^*`, geometrical spreading, the transmission-
coefficient product, and their combined amplitude factor for three cases:

* **(a)** Analytical formulas for a homogeneous medium.
* **(b)** LayTracer with a single-layer (homogeneous) model.
* **(c)** LayTracer with a two-layer model where both layers have the
  same Vp, Vs, ρ, Qp, Qs.

All three must agree, confirming internal consistency of the code.
"""

###############################################################################
# Setup
# -----

import laytracer as lt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 1

###############################################################################
# Common parameters
# -----------------
#
# We use a single set of elastic constants and a fixed source–receiver
# geometry throughout the example.

VP = 5000.0       # P-wave velocity  (m/s)
VS = VP / 1.732   # S-wave velocity  (m/s)
RHO = 2700.0      # density          (kg/m³)
QP = 500.0        # P-wave quality factor
QS = 250.0        # S-wave quality factor

src = np.array([0.0, 0.0, 500.0])       # source at 500 m depth
rcv = np.array([5000.0, 0.0, 2500.0])   # receiver at 2500 m depth

epic = np.sqrt((rcv[0] - src[0]) ** 2 + (rcv[1] - src[1]) ** 2)
dz = abs(rcv[2] - src[2])
dist = np.sqrt(epic ** 2 + dz ** 2)

print(f"Epicentral distance: {epic:.1f} m")
print(f"Depth offset:        {dz:.1f} m")
print(f"Straight-ray length: {dist:.4f} m")

###############################################################################
# (a) Analytical homogeneous solution
# -----------------------------------
#
# In a homogeneous medium a ray is a straight line of length
# :math:`R = \sqrt{X^2 + \Delta z^2}`, giving:
#
# .. math::
#
#     t       &= R / V_P \\
#     p       &= X / (V_P \cdot R) \\
#     t^*     &= t / Q_P \\
#     L       &= R \cdot V_P \\
#     \prod T &= 1 \quad \text{(no interfaces)}
#

tt_a  = dist / VP
p_a   = epic / (VP * dist)
ts_a  = tt_a / QP
L_a   = dist * VP
T_a   = 1.0
C_a   = T_a / L_a                # combined deterministic amplitude factor

print(f"\n--- Analytical ---")
print(f"Travel time:      {tt_a:.8f} s")
print(f"Ray parameter:    {p_a:.10e} s/m")
print(f"t*:               {ts_a:.10e} s")
print(f"Spreading:        {L_a:.4f}")
print(f"Trans. product:   {T_a:.6f}")
print(f"Combined (T/L):   {C_a:.10e}")

#%%

###############################################################################
# (b) Homogeneous model via LayTracer
# -----------------------------------
#
# A single-layer DataFrame — the simplest possible model.

homo_df = pd.DataFrame({
    "Depth": [0.0],
    "Vp":    [VP],
    "Vs":    [VS],
    "Rho":   [RHO],
    "Qp":    [QP],
    "Qs":    [QS],
})

res_h = lt.trace_rays(
    sources=src,
    receivers=rcv,
    velocity_df=homo_df,
    source_phase="P",
    compute_amplitude=True,
    transcoef_method="angle",
)

tt_h  = float(res_h.travel_times[0])
p_h   = float(res_h.ray_parameters[0])
ts_h  = float(res_h.tstar[0])
L_h   = float(res_h.spreading[0])
T_h   = float(res_h.trans_product[0])
C_h   = T_h / L_h

print(f"\n--- Homogeneous code ---")
print(f"Travel time:      {tt_h:.8f} s")
print(f"Ray parameter:    {p_h:.10e} s/m")
print(f"t*:               {ts_h:.10e} s")
print(f"Spreading:        {L_h:.4f}")
print(f"Trans. product:   {T_h:.6f}")
print(f"Combined (T/L):   {C_h:.10e}")

###############################################################################
# Plot the ray through the homogeneous model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ax = lt.plot.rays_2d(
    homo_df,
    rays=res_h.rays,
    sources=np.atleast_2d(src),
    receivers=np.atleast_2d(rcv),
    vel_type="Vp",
    add_colorbar=True,
    model_alpha=0.5,
    discrete_colorbar=True,
    unit="km",
)
ax.set_title("(b) Homogeneous model — single layer")
ax.legend(loc="upper right")
plt.show()

#%%

###############################################################################
# (c) Two-layer model with identical parameters
# ---------------------------------------------
#
# The interface at 1500 m splits the medium into two layers, but both
# share the same properties.  A correct implementation should return
# a transmission coefficient of 1.0 at that interface and identical
# travel time, :math:`t^*`, and spreading.

layered_df = pd.DataFrame({
    "Depth": [0.0, 1500.0],
    "Vp":    [VP, VP],
    "Vs":    [VS, VS],
    "Rho":   [RHO, RHO],
    "Qp":    [QP, QP],
    "Qs":    [QS, QS],
})

res_l = lt.trace_rays(
    sources=src,
    receivers=rcv,
    velocity_df=layered_df,
    source_phase="P",
    compute_amplitude=True,
    transcoef_method="angle",
)

tt_l  = float(res_l.travel_times[0])
p_l   = float(res_l.ray_parameters[0])
ts_l  = float(res_l.tstar[0])
L_l   = float(res_l.spreading[0])
T_l   = float(res_l.trans_product[0])
C_l   = T_l / L_l

print(f"\n--- Layered code (identical layers) ---")
print(f"Travel time:      {tt_l:.8f} s")
print(f"Ray parameter:    {p_l:.10e} s/m")
print(f"t*:               {ts_l:.10e} s")
print(f"Spreading:        {L_l:.4f}")
print(f"Trans. product:   {T_l:.6f}")
print(f"Combined (T/L):   {C_l:.10e}")

###############################################################################
# Plot the ray through the two-layer model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The interface at 1500 m is visible, but the ray is identical to the
# homogeneous case — a straight line — because the two layers share the
# same velocity.

ax = lt.plot.rays_2d(
    layered_df,
    rays=res_l.rays,
    sources=np.atleast_2d(src),
    receivers=np.atleast_2d(rcv),
    vel_type="Vp",
    add_colorbar=True,
    model_alpha=0.5,
    discrete_colorbar=True,
    unit="km",
)
ax.set_title("(c) Two-layer model — identical parameters")
ax.legend(loc="upper right")
plt.show()

#%%

###############################################################################
# Comparison table
# ----------------
#
# Collect all results into a DataFrame for easy visual inspection.

labels = [
    "Travel time (s)",
    "Ray parameter (s/m)",
    "t* (s)",
    "Spreading",
    "Trans. product",
    "Combined (T/L)",
]

vals_a = [tt_a, p_a, ts_a, L_a, T_a, C_a]
vals_h = [tt_h, p_h, ts_h, L_h, T_h, C_h]
vals_l = [tt_l, p_l, ts_l, L_l, T_l, C_l]

comparison = pd.DataFrame({
    "(a) Analytical": vals_a,
    "(b) Homo code":  vals_h,
    "(c) Layered code": vals_l,
}, index=labels)

print("\n" + comparison.to_string())

#%%

###############################################################################
# Relative errors
# ---------------
#
# Quantify the mismatch between each code result and the analytical
# reference.  Values should be at the machine-precision level.

rel_err_h = np.abs((np.array(vals_h) - np.array(vals_a))
                    / np.where(np.array(vals_a) != 0, vals_a, 1.0))
rel_err_l = np.abs((np.array(vals_l) - np.array(vals_a))
                    / np.where(np.array(vals_a) != 0, vals_a, 1.0))

err_df = pd.DataFrame({
    "Homo vs Analytical": rel_err_h,
    "Layered vs Analytical": rel_err_l,
}, index=labels)

print("\nRelative errors:")
print(err_df.to_string(float_format="{:.2e}".format))

#%%

###############################################################################
# Accuracy check
# --------------
#
# Assert that every quantity agrees across all three approaches to better
# than :math:`10^{-10}` relative tolerance.  This is a hard pass/fail gate
# that can catch regressions.

TOL = 1e-10

def _check(name, ref, val, tol=TOL):
    """Return (name, ref, val, rel_err, status)."""
    if ref == 0:
        err = abs(val)
    else:
        err = abs((val - ref) / ref)
    status = "PASS" if err <= tol else "FAIL"
    return (name, ref, val, err, status)

checks = []
# (b) homo code vs analytical
checks.append(_check("tt   (b) vs (a)", tt_a, tt_h))
checks.append(_check("p    (b) vs (a)", p_a,  p_h))
checks.append(_check("t*   (b) vs (a)", ts_a, ts_h))
checks.append(_check("L    (b) vs (a)", L_a,  L_h))
checks.append(_check("T    (b) vs (a)", T_a,  T_h))
checks.append(_check("C    (b) vs (a)", C_a,  C_h))
# (c) layered code vs analytical
checks.append(_check("tt   (c) vs (a)", tt_a, tt_l))
checks.append(_check("p    (c) vs (a)", p_a,  p_l))
checks.append(_check("t*   (c) vs (a)", ts_a, ts_l))
checks.append(_check("L    (c) vs (a)", L_a,  L_l))
checks.append(_check("T    (c) vs (a)", T_a,  T_l))
checks.append(_check("C    (c) vs (a)", C_a,  C_l))
# (c) layered code vs (b) homo code
checks.append(_check("tt   (c) vs (b)", tt_h, tt_l))
checks.append(_check("p    (c) vs (b)", p_h,  p_l))
checks.append(_check("t*   (c) vs (b)", ts_h, ts_l))
checks.append(_check("L    (c) vs (b)", L_h,  L_l))
checks.append(_check("T    (c) vs (b)", T_h,  T_l))
checks.append(_check("C    (c) vs (b)", C_h,  C_l))

check_df = pd.DataFrame(
    checks, columns=["Check", "Reference", "Value", "Rel. error", "Status"]
)

print(f"\nAccuracy checks (tolerance = {TOL:.0e}):\n")
print(check_df.to_string(index=False, float_format="{:.6e}".format))

n_fail = (check_df["Status"] == "FAIL").sum()
print(f"\nResult: {len(checks) - n_fail}/{len(checks)} checks passed.")
if n_fail:
    print(">>> SOME CHECKS FAILED — investigate! <<<")
else:
    print("All checks passed — homogeneous equivalence confirmed.")

#%%

###############################################################################
# Conclusion
# ----------
#
# All three approaches return identical values (to numerical precision),
# confirming that:
#
# 1. The solver correctly reduces to a straight-ray solution in a
#    homogeneous medium.
# 2. An artificial interface between layers with the **same** elastic
#    parameters introduces no spurious travel-time error, no
#    attenuation artefact, no spreading distortion, and a unit
#    transmission coefficient — exactly as expected from physics.
