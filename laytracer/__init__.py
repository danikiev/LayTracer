r"""
LayTracer — fast two-point ray tracing in 1-D layered media.

Public API
----------

Model
~~~~~
.. autosummary::
   LayerStack
   ModelArrays
   build_layer_stack

Solver
~~~~~~
.. autosummary::
   solve
   RayResult
   SolveDiagnostics

Multi-ray
~~~~~~~~~
.. autosummary::
   trace_rays
   TraceResult
   Interaction
   RayItinerary
   RaySensitivity

Approximation
~~~~~~~~~~~~~
.. autosummary::
   linearized_ray_change
   select_anchors
   LinearizedRayChange
   AnchorSelection
   TravelTimeApproximator
   TravelTimePrediction

Amplitude
~~~~~~~~~
.. autosummary::
   transmission_normal
   sh_rt_coefficients
   psv_rt_coefficients
   critical_angle
   find_critical_angles
   find_brewster_angles
   normalize_rt_coefficient

Visualisation
~~~~~~~~~~~~~
.. autosummary::
   plot
"""

from .model import LayerStack, ModelArrays, build_layer_stack
from .solver import (
    RayResult,
    SolveDiagnostics,
    solve,
    offset,
    offset_dq,
    offset_dq2,
    q_from_p,
    p_from_q,
    initial_q,
    newton_step,
)
from .amplitude import (
    transmission_normal,
    sh_rt_coefficients,
    psv_rt_coefficients,
    critical_angle,
    find_critical_angles,
    find_brewster_angles,
    normalize_rt_coefficient,
)
from .api import Interaction, RayItinerary, TraceResult, trace_rays
from .sensitivity import RaySensitivity
from .approximation import (
    AnchorSelection,
    LinearizedRayChange,
    TravelTimeApproximator,
    TravelTimePrediction,
    linearized_ray_change,
    select_anchors,
)
from . import plot

try:
    from .version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    # model
    "LayerStack",
    "ModelArrays",
    "build_layer_stack",
    # solver
    "RayResult",
    "SolveDiagnostics",
    "solve",
    "offset",
    "offset_dq",
    "offset_dq2",
    "q_from_p",
    "p_from_q",
    "initial_q",
    "newton_step",
    # amplitude
    "transmission_normal",
    "sh_rt_coefficients",
    "psv_rt_coefficients",
    "critical_angle",
    "find_critical_angles",
    "find_brewster_angles",
    "normalize_rt_coefficient",
    # api
    "TraceResult",
    "Interaction",
    "RayItinerary",
    "RaySensitivity",
    "trace_rays",
    # approximation
    "AnchorSelection",
    "LinearizedRayChange",
    "TravelTimeApproximator",
    "TravelTimePrediction",
    "linearized_ray_change",
    "select_anchors",
    # visualisation
    "plot",
]
