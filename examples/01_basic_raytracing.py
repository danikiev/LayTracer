r"""
Basic ray tracing in a layered model
=====================================

This example demonstrates two-point ray tracing through a simple 3-layer
velocity model using LayTracer.
"""

###############################################################################
# Setup
# -----

import numpy as np
import pandas as pd
import laytracer
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 2

###############################################################################
# Define velocity model
# ---------------------
#
# A 3-layer model with increasing velocity with depth.

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
# Plot velocity profile
# ---------------------

ax = laytracer.plot.velocity_profile(vel_df, vel_type="Vp")

#%%

###############################################################################
# Trace a single ray in 2-D
# -------------------------
#
# Trace a P-wave from a source at depth 3000 m to a receiver at the surface.

stack = laytracer.build_layer_stack(vel_df, z_src=3000.0, z_rcv=0.0)

res = laytracer.solve(
    stack,
    epicentral_dist=5000.0,
    z_src=3000.0,
    z_rcv=0.0,
    vel_type="Vp",
)

print(f"Travel time:    {res.travel_time:.4f} s")
print(f"Ray parameter:  {res.ray_parameter:.6e} s/m")

###############################################################################
# Plot the 2-D ray
# ----------------

ax = laytracer.plot.rays_2d(
    vel_df,
    rays=[res.ray_path],
    sources=np.array([[0.0, 0.0, 3000.0]]),
    receivers=np.array([[5000.0, 0.0, 0.0]]),
    vel_type="Vp",
)

#%%

###############################################################################
# Trace multiple rays in 3-D
# --------------------------
#
# Use :func:`laytracer.trace_rays` to trace from one source to multiple
# receivers arranged in a circle.

src = np.array([0.0, 0.0, 3000.0])

# Receivers on surface in a circle of radius 5000 m
n_rcv = 12
angles = np.linspace(0, 2 * np.pi, n_rcv, endpoint=False)
rcvs = np.column_stack([
    5000.0 * np.cos(angles),
    5000.0 * np.sin(angles),
    np.zeros(n_rcv),
])

result = laytracer.trace_rays(
    sources=src,
    receivers=rcvs,
    velocity_df=vel_df,
    vel_type="Vp",
)

print(f"Number of rays: {len(result.rays)}")
print(f"Travel times:   {result.travel_times}")

###############################################################################
# Plot 3-D rays
# -------------

fig = laytracer.plot.rays_3d(
    vel_df,
    rays=result.rays,
    sources=src,
    receivers=rcvs,
)
fig

###############################################################################
# Show all plots when running from the command line.

plt.show()
