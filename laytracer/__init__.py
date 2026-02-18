r"""
LayTracer — fast two-point ray tracing in 1-D layered media.

Public API
----------

Model
~~~~~
.. autosummary::
   LayerStack
   build_layer_stack

Solver
~~~~~~
.. autosummary::
   solve
   RayResult

Multi-ray
~~~~~~~~~
.. autosummary::
   trace_rays
   TraceResult

Amplitude
~~~~~~~~~
.. autosummary::
   transmission_normal
   psv_rt_coefficients
   find_brewster_angles

Visualisation
~~~~~~~~~~~~~
.. autosummary::
   plot
"""

from .model import LayerStack, build_layer_stack
from .solver import (
    RayResult,
    solve,
    offset,
    offset_dq,
    offset_dq2,
    q_from_p,
    p_from_q,
    initial_q,
    newton_step,
)
from .amplitude import transmission_normal, psv_rt_coefficients, find_brewster_angles
from .api import TraceResult, trace_rays
from . import plot

try:
    from .version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    # model
    "LayerStack",
    "build_layer_stack",
    # solver
    "RayResult",
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
    "psv_rt_coefficients",
    "find_brewster_angles",
    # api
    "TraceResult",
    "trace_rays",
    # visualisation
    "plot",
]
