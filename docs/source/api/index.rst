.. _api:

=============
API Reference
=============

.. automodule:: laytracer
   :no-members:

Model
-----

.. autoclass:: laytracer.LayerStack
   :members:

.. autoclass:: laytracer.ModelArrays
   :members:

.. autofunction:: laytracer.build_layer_stack

Solver
------

.. autofunction:: laytracer.solve

.. autoclass:: laytracer.RayResult
   :members:

.. autoclass:: laytracer.SolveDiagnostics
   :members:

.. autofunction:: laytracer.offset

.. autofunction:: laytracer.offset_dq

.. autofunction:: laytracer.offset_dq2

.. autofunction:: laytracer.q_from_p

.. autofunction:: laytracer.p_from_q

.. autofunction:: laytracer.initial_q

.. autofunction:: laytracer.newton_step

Multi-ray interface
-------------------

.. autofunction:: laytracer.trace_rays

.. autoclass:: laytracer.TraceResult
   :members:

.. autoclass:: laytracer.Interaction
   :members:

.. autoclass:: laytracer.RayItinerary
   :members:

.. autoclass:: laytracer.RaySensitivity
   :members:

Traveltime approximation
------------------------

.. autofunction:: laytracer.linearized_ray_change

.. autofunction:: laytracer.select_anchors

.. autoclass:: laytracer.LinearizedRayChange
   :members:

.. autoclass:: laytracer.AnchorSelection
   :members:

.. autoclass:: laytracer.TravelTimeApproximator
   :members:

.. autoclass:: laytracer.TravelTimePrediction
   :members:

Amplitude
---------

.. autofunction:: laytracer.transmission_normal

.. autofunction:: laytracer.sh_rt_coefficients

.. autofunction:: laytracer.psv_rt_coefficients

.. autofunction:: laytracer.critical_angle

.. autofunction:: laytracer.find_critical_angles

.. autofunction:: laytracer.normalize_rt_coefficient

.. autofunction:: laytracer.find_brewster_angles

Visualisation
-------------

.. autofunction:: laytracer.plot.coefficient_panels

.. autofunction:: laytracer.plot.velocity_profile

.. autofunction:: laytracer.plot.rays_2d

.. autofunction:: laytracer.plot.rays_3d

.. only:: latex

   .. bibliography::
      :style: unsrt
