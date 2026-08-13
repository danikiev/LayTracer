r"""
06. See what a ray is sensitive to
==================================

One traced ray can say more than where the ray goes and when it arrives.
LayTracer can also report how its travel time and physical ray parameter
respond to small changes in the model and acquisition geometry.

This example traces a P wave that reflects as SV.  The downgoing and upgoing
legs therefore sample different velocity fields, giving one compact example
with non-zero derivatives for :math:`V_P`, :math:`V_S`, interface depths, and
the source and receiver coordinates.
"""

###############################################################################
# Trace one selected ray
# ----------------------
#
# The source and receiver are 200 m deep.  The selected wave travels down as P,
# reflects at 2500 m, and returns as SV.  Requesting ``sensitivities`` and
# ``diagnostics`` is opt-in; the ordinary travel-time API is unchanged.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import laytracer as lt


# sphinx_gallery_thumbnail_number = 1

model = pd.DataFrame({
    "Depth": [0.0, 1200.0, 2500.0],
    "Vp": [2400.0, 3600.0, 5200.0],
    "Vs": [1350.0, 2050.0, 3000.0],
})
source = np.array([0.0, 0.0, 200.0])
receiver = np.array([4000.0, 0.0, 200.0])


def trace_ps_reflection(model_to_trace, source_to_trace, receiver_to_trace, requested):
    """Trace the selected P-down, SV-up reflection."""
    reflector_depth = float(model_to_trace.iloc[2]["Depth"])
    itinerary = lt.RayItinerary(
        source_phase="P",
        interactions=[lt.Interaction(reflector_depth, "reflect", "SV")],
    )
    return lt.trace_rays(
        source_to_trace,
        receiver_to_trace,
        model_to_trace,
        itinerary=itinerary,
        requested=requested,
        n_jobs=1,
        verbose=False,
        tol=1e-10,
        max_iter=30,
    )


reference = trace_ps_reflection(
    model,
    source,
    receiver,
    {"travel_times", "rays", "ray_parameters", "diagnostics", "sensitivities"},
)
diagnostic = reference.diagnostics[0]
sensitivity = reference.sensitivities[0]

assert diagnostic.converged
assert sensitivity.valid
np.testing.assert_array_equal(sensitivity.vp_layer_indices, [0, 1])
np.testing.assert_array_equal(sensitivity.vs_layer_indices, [0, 1])
np.testing.assert_array_equal(sensitivity.interface_indices, [1, 2])
assert sensitivity.dtravel_time_dsource[1] == 0.0
assert sensitivity.dtravel_time_dreceiver[1] == 0.0

print(f"Travel time: {reference.travel_times[0]:.4f} s")
print(f"Physical ray parameter p: {1e6 * reference.ray_parameters[0]:.3f} us/m")
print(
    f"Solve: {diagnostic.method}, endpoint residual "
    f"{diagnostic.absolute_offset_residual:.2e} m"
)
print("Vp layer indices:", sensitivity.vp_layer_indices)
print("dT/dVp:", sensitivity.dtravel_time_dvp)
print("dp/dVp:", sensitivity.dray_parameter_dvp)
print("Vs layer indices:", sensitivity.vs_layer_indices)
print("dT/dVs:", sensitivity.dtravel_time_dvs)
print("dp/dVs:", sensitivity.dray_parameter_dvs)
print("Interface indices:", sensitivity.interface_indices)
print("dT/dz_interface:", sensitivity.dtravel_time_dinterface_depths)
print("dp/dz_interface:", sensitivity.dray_parameter_dinterface_depths)
print("dT/dsource:", sensitivity.dtravel_time_dsource)
print("dp/dsource:", sensitivity.dray_parameter_dsource)
print("dT/dreceiver:", sensitivity.dtravel_time_dreceiver)
print("dp/dreceiver:", sensitivity.dray_parameter_dreceiver)

###############################################################################
# Visualize the sensitivity fingerprint
# -------------------------------------
#
# Raw derivatives have different units, so plotting them together would be
# misleading.  Instead, apply small, clearly labelled changes: 1% to a
# velocity and 10 m to an interface or endpoint.  Each derivative then becomes
# a predicted travel-time shift in milliseconds and a physical-ray-parameter
# shift in microseconds per metre.

effect_rows = []

for layer_index, dtime, dparameter in zip(
    sensitivity.vp_layer_indices,
    sensitivity.dtravel_time_dvp,
    sensitivity.dray_parameter_dvp,
):
    change = 0.01 * model.loc[layer_index, "Vp"]
    effect_rows.append((
        rf"$V_P$ layer {layer_index + 1} (+1%)",
        "P velocity",
        dtime * change,
        dparameter * change,
    ))

for layer_index, dtime, dparameter in zip(
    sensitivity.vs_layer_indices,
    sensitivity.dtravel_time_dvs,
    sensitivity.dray_parameter_dvs,
):
    change = 0.01 * model.loc[layer_index, "Vs"]
    effect_rows.append((
        rf"$V_S$ layer {layer_index + 1} (+1%)",
        "S velocity",
        dtime * change,
        dparameter * change,
    ))

for interface_index, dtime, dparameter in zip(
    sensitivity.interface_indices,
    sensitivity.dtravel_time_dinterface_depths,
    sensitivity.dray_parameter_dinterface_depths,
):
    label = "reflector" if interface_index == 2 else "interface 1"
    effect_rows.append((
        f"{label} deeper (+10 m)",
        "Interface",
        dtime * 10.0,
        dparameter * 10.0,
    ))

endpoint_changes = (
    ("source right (+10 m)", "Source", sensitivity.dtravel_time_dsource[0], sensitivity.dray_parameter_dsource[0]),
    ("source deeper (+10 m)", "Source", sensitivity.dtravel_time_dsource[2], sensitivity.dray_parameter_dsource[2]),
    ("receiver right (+10 m)", "Receiver", sensitivity.dtravel_time_dreceiver[0], sensitivity.dray_parameter_dreceiver[0]),
    ("receiver deeper (+10 m)", "Receiver", sensitivity.dtravel_time_dreceiver[2], sensitivity.dray_parameter_dreceiver[2]),
)
for label, group, dtime, dparameter in endpoint_changes:
    effect_rows.append((label, group, dtime * 10.0, dparameter * 10.0))

effects = pd.DataFrame(
    effect_rows,
    columns=["change", "group", "delta_time_s", "delta_parameter_s_per_m"],
)
effects["delta_time_ms"] = 1e3 * effects["delta_time_s"]
effects["delta_parameter_us_per_m"] = 1e6 * effects["delta_parameter_s_per_m"]

print("\nSensitivity fingerprint for the labelled changes:")
print(effects[["change", "delta_time_ms", "delta_parameter_us_per_m"]].to_string(index=False))

# Move only the reflector for the dashed comparison ray.
deeper_model = model.copy()
deeper_model.loc[2, "Depth"] += 100.0
deeper = trace_ps_reflection(
    deeper_model,
    source,
    receiver,
    {"travel_times", "rays", "ray_parameters"},
)

group_colors = {
    "P velocity": "#0072B2",
    "S velocity": "#CC79A7",
    "Interface": "#009E73",
    "Source": "#E69F00",
    "Receiver": "#6B6B6B",
}
colors = [group_colors[group] for group in effects["group"]]

figure, axes = plt.subplots(
    1,
    3,
    figsize=(16.0, 5.8),
    gridspec_kw={"width_ratios": [1.1, 1.35, 1.35]},
)

axis = axes[0]
axis.axhspan(0.0, 1.2, color="#DCEAF4")
axis.axhspan(1.2, 2.5, color="#B7D5E8")
axis.axhspan(2.5, 2.9, color="#E9ECEF")
axis.axhline(1.2, color="white", linewidth=1.4)
axis.axhline(2.5, color="#009E73", linewidth=1.3)
axis.axhline(2.6, color="#009E73", linewidth=1.3, linestyle="--")

ray = reference.rays[0]
turning_index = int(np.argmax(ray[:, 2]))
axis.plot(ray[: turning_index + 1, 0] / 1000.0, ray[: turning_index + 1, 2] / 1000.0, color="#0072B2", linewidth=2.5)
axis.plot(ray[turning_index:, 0] / 1000.0, ray[turning_index:, 2] / 1000.0, color="#CC79A7", linewidth=2.5)
perturbed_ray = deeper.rays[0]
axis.plot(perturbed_ray[:, 0] / 1000.0, perturbed_ray[:, 2] / 1000.0, color="#333333", linewidth=1.3, linestyle="--", alpha=0.8)
axis.scatter(source[0] / 1000.0, source[2] / 1000.0, marker="*", s=90, color="#222222", zorder=5)
axis.scatter(receiver[0] / 1000.0, receiver[2] / 1000.0, marker="v", s=55, color="#222222", zorder=5)
axis.text(0.12, 0.78, "layer 1", transform=axis.transAxes, color="#354F60")
axis.text(0.12, 0.47, "layer 2", transform=axis.transAxes, color="#354F60")
axis.text(2.65, 2.78, "layer below reflector\n(not sampled)", ha="center", color="#6B6B6B", fontsize=8)
axis.text(0.58, 1.02, "P", color="#0072B2", fontweight="bold")
axis.text(3.55, 0.72, "SV", color="#CC79A7", fontweight="bold")
axis.text(0.12, 2.47, "reference reflector", color="#007C5B", va="bottom", fontsize=8)
axis.text(0.12, 2.67, "reflector +100 m", color="#007C5B", va="top", fontsize=8)
axis.text(
    0.52,
    0.965,
    "$\\Delta T$: arrival shift     $\\Delta p$: launch-direction shift",
    transform=axis.transAxes,
    ha="center",
    va="top",
    fontsize=8.0,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#BBBBBB"},
)
axis.set_xlim(-0.15, 4.15)
axis.set_ylim(2.9, 0.0)
axis.set_xlabel("Horizontal distance (km)")
axis.set_ylabel("Depth (km)")
axis.set_title("(a) One selected P-to-SV reflection")


def sensitivity_lollipops(axis, values, xlabel, title):
    """Draw one signed, annotated sensitivity-effect chart."""
    positions = np.arange(len(effects))
    limit = 1.28 * np.max(np.abs(values))
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.hlines(positions, 0.0, values, colors=colors, linewidth=2.0)
    axis.scatter(values, positions, c=colors, s=43, zorder=3)
    for position, value in zip(positions, values):
        offset = 0.025 * limit if value >= 0.0 else -0.025 * limit
        axis.text(
            value + offset,
            position,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0.0 else "right",
            fontsize=7.4,
        )
    axis.set_xlim(-limit, limit)
    axis.set_ylim(len(effects) - 0.35, -0.65)
    axis.set_yticks(positions, effects["change"])
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.22)


sensitivity_lollipops(
    axes[1],
    effects["delta_time_ms"].to_numpy(),
    "Predicted arrival-time change (ms)",
    "(b) What moves the arrival time?",
)
sensitivity_lollipops(
    axes[2],
    effects["delta_parameter_us_per_m"].to_numpy(),
    r"Predicted change in physical $p$ ($\mu$s/m)",
    "(c) What steers the ray?",
)
axes[2].set_yticklabels([])

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", color=color, label=group)
    for group, color in group_colors.items()
]
figure.legend(handles=legend_handles, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.67, 0.01))
figure.suptitle("What changes this arrival?", fontsize=14)
figure.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.18, wspace=0.58)
plt.show()

###############################################################################
# Use the derivatives as a local predictor
# ----------------------------------------
#
# A model update usually changes several parameters at once.  The predicted
# change is obtained by multiplying each perturbation by its derivative and
# adding the contributions.  The random nearby models below are retraced only
# to verify that one reference sensitivity record predicts both outputs.


def perturb_and_predict(scale, random_generator):
    """Perturb all active parameter classes and compare with a retrace."""
    trial_model = model.copy()
    trial_source = source.copy()
    trial_receiver = receiver.copy()
    predicted_time = 0.0
    predicted_parameter = 0.0

    delta_vp = random_generator.normal(size=2) * 0.01 * scale * model.loc[[0, 1], "Vp"].to_numpy()
    delta_vs = random_generator.normal(size=2) * 0.01 * scale * model.loc[[0, 1], "Vs"].to_numpy()
    delta_interfaces = random_generator.normal(size=2) * 10.0 * scale
    delta_source = random_generator.normal(size=2) * 10.0 * scale
    delta_receiver = random_generator.normal(size=2) * 10.0 * scale

    trial_model.loc[[0, 1], "Vp"] += delta_vp
    trial_model.loc[[0, 1], "Vs"] += delta_vs
    trial_model.loc[[1, 2], "Depth"] += delta_interfaces
    trial_source[[0, 2]] += delta_source
    trial_receiver[[0, 2]] += delta_receiver

    predicted_time += np.dot(sensitivity.dtravel_time_dvp, delta_vp)
    predicted_time += np.dot(sensitivity.dtravel_time_dvs, delta_vs)
    predicted_time += np.dot(sensitivity.dtravel_time_dinterface_depths, delta_interfaces)
    predicted_time += np.dot(sensitivity.dtravel_time_dsource[[0, 2]], delta_source)
    predicted_time += np.dot(sensitivity.dtravel_time_dreceiver[[0, 2]], delta_receiver)

    predicted_parameter += np.dot(sensitivity.dray_parameter_dvp, delta_vp)
    predicted_parameter += np.dot(sensitivity.dray_parameter_dvs, delta_vs)
    predicted_parameter += np.dot(sensitivity.dray_parameter_dinterface_depths, delta_interfaces)
    predicted_parameter += np.dot(sensitivity.dray_parameter_dsource[[0, 2]], delta_source)
    predicted_parameter += np.dot(sensitivity.dray_parameter_dreceiver[[0, 2]], delta_receiver)

    assert np.all(np.diff(trial_model["Depth"]) > 0.0)
    trial = trace_ps_reflection(
        trial_model,
        trial_source,
        trial_receiver,
        {"travel_times", "ray_parameters"},
    )
    exact_time = trial.travel_times[0] - reference.travel_times[0]
    exact_parameter = trial.ray_parameters[0] - reference.ray_parameters[0]
    return predicted_time, exact_time, predicted_parameter, exact_parameter


scales = np.array([0.25, 1.0, 4.0])
comparisons = {}
relative_time_error = []
relative_parameter_error = []

for scale in scales:
    # Reuse the same perturbation directions at every scale so that this
    # comparison isolates the effect of moving farther from the reference.
    scale_rng = np.random.default_rng(7)
    values = np.array([perturb_and_predict(scale, scale_rng) for _ in range(80)])
    comparisons[scale] = values
    predicted_time, exact_time, predicted_parameter, exact_parameter = values.T
    relative_time_error.append(
        100.0 * np.sqrt(np.mean((predicted_time - exact_time) ** 2))
        / np.sqrt(np.mean(exact_time**2))
    )
    relative_parameter_error.append(
        100.0 * np.sqrt(np.mean((predicted_parameter - exact_parameter) ** 2))
        / np.sqrt(np.mean(exact_parameter**2))
    )

nominal_index = int(np.flatnonzero(scales == 1.0)[0])
assert relative_time_error[nominal_index] < 5.0
assert relative_parameter_error[nominal_index] < 5.0

print("\nRelative RMS prediction error:")
for scale, time_error, parameter_error in zip(scales, relative_time_error, relative_parameter_error):
    print(f"  {scale:g}x: travel time {time_error:.2f}%, ray parameter {parameter_error:.2f}%")

nominal = comparisons[1.0]
predicted_time, exact_time, predicted_parameter, exact_parameter = nominal.T
figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))


def prediction_panel(axis, exact, predicted, scale_factor, units, title):
    """Plot analytic predictions against independent retraces."""
    exact_scaled = scale_factor * exact
    predicted_scaled = scale_factor * predicted
    lower = min(exact_scaled.min(), predicted_scaled.min())
    upper = max(exact_scaled.max(), predicted_scaled.max())
    padding = 0.07 * (upper - lower)
    bounds = [lower - padding, upper + padding]
    axis.scatter(exact_scaled, predicted_scaled, s=24, color="#0072B2", alpha=0.65, edgecolors="none")
    axis.plot(bounds, bounds, color="#333333", linestyle="--", linewidth=1.0, label="perfect prediction")
    axis.set_xlim(bounds)
    axis.set_ylim(bounds)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"Retraced change ({units})")
    axis.set_ylabel(f"Derivative prediction ({units})")
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.2)


prediction_panel(
    axes[0],
    exact_time,
    predicted_time,
    1e3,
    "ms",
    "(a) Arrival-time changes",
)
prediction_panel(
    axes[1],
    exact_parameter,
    predicted_parameter,
    1e6,
    r"$\mu$s/m",
    r"(b) Physical $p$ changes",
)

axis = axes[2]
axis.plot(scales, relative_time_error, "o-", color="#0072B2", label="travel time")
axis.plot(scales, relative_parameter_error, "s-", color="#CC79A7", label="physical ray parameter")
axis.set_xscale("log", base=2)
axis.set_xticks(scales, [r"$\frac{1}{4}\times$", r"$1\times$", r"$4\times$"])
axis.set_xlabel("Perturbation size")
axis.set_ylabel("RMS prediction error\n(% of the exact change)")
axis.set_title("(c) Where is the local prediction useful?")
axis.legend(frameon=False, fontsize=8)
axis.grid(alpha=0.2)

active_derivatives = len(effects)
finite_difference_traces = 2 * active_derivatives + 1
figure.text(
    0.5,
    0.025,
    f"1 reference trace returns {active_derivatives} active parameter sensitivities; "
    f"a centered finite-difference Jacobian would require {finite_difference_traces} traces.",
    ha="center",
    va="bottom",
    fontsize=9.5,
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F2F6F8", "edgecolor": "#AAB7BF"},
)
figure.suptitle("One traced ray predicts nearby rays", fontsize=14)
figure.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.25, wspace=0.34)
plt.show()

###############################################################################
# The analytic derivatives are local: the prediction is best for small changes
# that preserve the endpoint layers, selected P-to-SV itinerary, and
# propagating branch.  That is the same regime used by gradient-based
# inversion, uncertainty propagation, and rapid evaluation of nearby models.
