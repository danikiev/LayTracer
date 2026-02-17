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

.. autofunction:: laytracer.build_layer_stack

Solver
------

.. autofunction:: laytracer.solve

.. autoclass:: laytracer.RayResult
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

Amplitude
---------

.. autofunction:: laytracer.transmission_normal

.. autofunction:: laytracer.transmission_psv

Visualisation
-------------

.. autofunction:: laytracer.plot.velocity_profile

.. autofunction:: laytracer.plot.rays_2d

.. autofunction:: laytracer.plot.rays_3d
