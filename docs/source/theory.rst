.. _theory:

======
Theory
======

This chapter presents the theoretical foundations of the ray tracing
algorithm implemented in LayTracer.

.. contents:: Contents
   :local:
   :depth: 2

----

Two-point ray tracing problem
=============================

Given a horizontally layered velocity model with :math:`N` constant-velocity
layers and two points—a source S at :math:`(x_s, z_s)` and a receiver R at
:math:`(x_r, z_r)`—the task is to find the ray path connecting them that
satisfies **Snell's law** at every interface:

.. math::
   p = \frac{\sin \theta_k}{v_k} = \text{const}

where :math:`p` is the **horizontal slowness** (ray parameter) and :math:`\theta_k`
is the ray angle from vertical in layer :math:`k`.

----

Dimensionless ray parameter
============================

:footcite:t:`FangChen2019` introduce a dimensionless parameter

.. math::
   q = \frac{p\,v_{\max}}{\sqrt{1 - p^2\,v_{\max}^2}}

where :math:`v_{\max} = \max_k v_k`.  The inverse relation is

.. math::
   p = \frac{q}{v_{\max}\,\sqrt{1 + q^2}}

Key advantages:

- :math:`q \in [0, \infty)` maps the full range of valid take-off angles
- The offset function :math:`X(q)` is well-behaved (monotonically increasing,
  smooth, avoids singularities near the critical angle)

----

Offset equation
===============

The total horizontal distance (offset) :math:`X` traversed by a ray through
:math:`N` layers is (Eq. 3 of :footcite:t:`FangChen2019`):

.. math::
   X(q) = \sum_{k=1}^{N} \frac{q\,\lambda_k\,h_k}
          {\sqrt{1 + (1 - \lambda_k^2)\,q^2}}

where :math:`\lambda_k = v_k / v_{\max}` and :math:`h_k` is the thickness of the
:math:`k`-th traversed layer.

First derivative
----------------

.. math::
   \frac{\mathrm{d}X}{\mathrm{d}q}
   = \sum_{k=1}^{N} \frac{\lambda_k\,h_k}
     {\bigl[1 + (1 - \lambda_k^2)\,q^2\bigr]^{3/2}}

Second derivative
-----------------

.. math::
   \frac{\mathrm{d}^2X}{\mathrm{d}q^2}
   = -3\,q\,\sum_{k=1}^{N}
     \frac{(1 - \lambda_k^2)\,\lambda_k\,h_k}
     {\bigl[1 + (1 - \lambda_k^2)\,q^2\bigr]^{5/2}}

----

Asymptotic initial estimate
============================

Two linear asymptotes of :math:`X(q)` provide an efficient initial guess
(Section 2.2 of :footcite:t:`FangChen2019`):

**Near-field** (:math:`q \to 0`):

.. math::
   X \approx m_0\,q, \qquad m_0 = \sum_k \lambda_k\,h_k

**Far-field** (:math:`q \to \infty`):

.. math::
   X \approx m_\infty\,q + b_\infty

where

.. math::
   m_\infty = \sum_{k:\,\lambda_k=1} h_k, \qquad
   b_\infty = \sum_{k:\,\lambda_k<1}
              \frac{\lambda_k\,h_k}{\sqrt{1 - \lambda_k^2}}

The initial estimate :math:`q_0` is chosen from the appropriate asymptote
based on whether the target offset :math:`X_R` falls in the near-field
or far-field regime.

----

Quadratic Newton iteration
===========================

The two-point problem :math:`X(q) = X_R` is solved by second-order
Newton iteration (Section 2.3 of :footcite:t:`FangChen2019`).  At each
step, :math:`X(q)` is expanded to second order about the current iterate
:math:`q_i`:

.. math::
   \tfrac{1}{2}\,X''(q_i)\,\Delta q^2 + X'(q_i)\,\Delta q
   + \bigl[X(q_i) - X_R\bigr] = 0

This quadratic equation in :math:`\Delta q` yields two roots.  The root
minimising :math:`|X(q_i + \Delta q) - X_R|` is selected.  Convergence
is typically achieved within **2–3 iterations**.

----

Travel time
===========

Once the ray parameter :math:`p` is determined, the travel time through
each layer is

.. math::
   \Delta t_k = \frac{h_k}{v_k^2\,\eta_k}

where :math:`\eta_k = \sqrt{1/v_k^2 - p^2}` is the vertical slowness in
layer :math:`k`.  The total travel time is :math:`t = \sum_k \Delta t_k`.

----

Attenuation operator :math:`t^*`
=================================

The attenuation operator (:footcite:t:`AkiRichards2002`, Ch. 5) measures
the integrated effect of intrinsic absorption along the ray path:

.. math::
   t^* = \sum_{k=1}^{N} \frac{\Delta t_k}{Q_k}

where :math:`Q_k` is the quality factor in layer :math:`k`.  This gives
the spectral decay :math:`\exp(-\pi f\,t^*)` at frequency :math:`f`.

For a **vertical ray**: :math:`t^* = \sum h_k / (v_k\,Q_k)`.

For **uniform** :math:`Q`: :math:`t^* = t / Q`.

----

Geometrical spreading
=====================

In a 1-D layered medium with cylindrical symmetry (3-D point source),
the geometrical spreading factor :math:`L` depends on the **ray tube
Jacobian** :math:`\partial X / \partial p`
(:footcite:t:`Cerveny2001`; :footcite:t:`AkiRichards2002`, Ch. 4):

.. math::
   L = \sqrt{\frac{X \cdot |\partial X / \partial p|}
                  {\cos\theta_s \cdot \cos\theta_r}}

where :math:`\theta_s, \theta_r` are the ray angles at source and
receiver.  The derivative :math:`\partial X / \partial p` is computed
analytically via the chain rule:

.. math::
   \frac{\partial X}{\partial p}
   = \frac{\mathrm{d}X}{\mathrm{d}q} \cdot \frac{\mathrm{d}q}{\mathrm{d}p},
   \qquad
   \frac{\mathrm{d}q}{\mathrm{d}p}
   = \frac{v_{\max}}{(1 - p^2\,v_{\max}^2)^{3/2}}

----

Transmission coefficients
=========================

Normal incidence
----------------

The displacement amplitude transmission coefficient at normal incidence
(:footcite:t:`Shearer2019`) is

.. math::
   T = \frac{2\,Z_1}{Z_1 + Z_2}, \qquad Z_i = \rho_i\,v_i

Angle-dependent (Zoeppritz)
---------------------------

For oblique incidence, LayTracer solves the full 4×4 Zoeppritz system
(:footcite:t:`AkiRichards2002`, Eqs. 5.40–5.42).  The system relates
the reflected and transmitted P-SV amplitudes to the incident
wave amplitude through displacement and stress continuity conditions
at each interface.

----

References
==========

.. footbibliography::
