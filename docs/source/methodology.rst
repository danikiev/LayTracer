.. _methodology:

===========
Methodology
===========

This chapter presents the theoretical foundations of the ray tracing
algorithm implemented in LayTracer.

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

Extension to 3-D layered media
===============================

All the formulae above are derived in a **2-D vertical plane**
containing both the source and the receiver.  Because horizontally
layered media possess *cylindrical symmetry* about the vertical axis
through the source, any source–receiver pair in three-dimensional space
can be **reduced to an equivalent 2-D problem** without loss of
generality (:footcite:t:`Cerveny2001`, Ch. 3;
:footcite:t:`AkiRichards2002`, Ch. 4).

Coordinate projection
---------------------

Let the source position be :math:`\mathbf{s} = (x_s, y_s, z_s)` and the
receiver position :math:`\mathbf{r} = (x_r, y_r, z_r)`.  The
**epicentral distance** (horizontal distance) is

.. math::
   \Delta = \sqrt{(x_r - x_s)^2 + (y_r - y_s)^2}

and the **unit direction vector** from source to receiver in the
horizontal plane is

.. math::
   \hat{\mathbf{u}} = \frac{1}{\Delta}
       \begin{pmatrix} x_r - x_s \\ y_r - y_s \end{pmatrix},
   \qquad \Delta > 0

(for vanishing :math:`\Delta` the azimuth is arbitrary, and we default to
:math:`\hat{\mathbf{u}} = (1,\,0)`).

The 2-D ray tracing problem is then solved exactly as described in the
preceding sections, with the epicentral distance :math:`\Delta` playing
the role of the target offset :math:`X_R` and the depth coordinates
:math:`z_s, z_r` determining the traversed layers.

Back-projection to 3-D
-----------------------

Once the 2-D ray path :math:`\{(x_k^{(2\mathrm{D})},\, z_k)\}_{k=0}^M`
has been computed, it is mapped back to 3-D Cartesian coordinates via

.. math::
   \begin{aligned}
   X_k &= x_s + x_k^{(2\mathrm{D})}\;\hat{u}_x, \\
   Y_k &= y_s + x_k^{(2\mathrm{D})}\;\hat{u}_y, \\
   Z_k &= z_k.
   \end{aligned}

This simply *sweeps* the 2-D ray along the source–receiver azimuth.

Validity of amplitude attributes
---------------------------------

Because the medium properties depend only on depth, every quantity
computed from the 2-D ray remains valid in 3-D:

- **Ray parameter** :math:`p` — determined solely by Snell's law across
  horizontal interfaces.
- **Travel time** — sum of vertical-slowness contributions
  :math:`\Delta t_k`, unchanged by azimuth.
- **Attenuation operator** :math:`t^*` — depends only on
  :math:`\Delta t_k` and :math:`Q_k`.
- **Geometrical spreading** — the formula
  :math:`L = \sqrt{X\,|\partial X/\partial p|\,/\,
  (\cos\theta_s\cos\theta_r)}` already accounts for 3-D cylindrical
  divergence because it includes the factor :math:`X` (epicentral
  distance) that captures the out-of-plane ray-tube expansion
  (:footcite:t:`Cerveny2001`, §4.10).
- **Transmission coefficients** — depend on ray parameter and layer
  impedances, not on azimuth.

Thus, the layered-media solver need only operate in the 2-D ray plane;
the full 3-D solution is recovered by geometry alone.

----

References
==========

.. footbibliography::
