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
   q = \sqrt{\frac{p^2}{1/v_{\max}^2 - p^2}}
     = \frac{p\,v_{\max}}{\sqrt{1 - p^2\,v_{\max}^2}}

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
(see "Initial estimate of q" in :footcite:t:`FangChen2019`):

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
Newton iteration (see :footcite:t:`FangChen2019`).  At each
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
the cumulative dissipative loss of wave amplitude along the ray path. Since the
spatial path length in layer :math:`k` is :math:`\Delta s_k = v_k \Delta t_k`,
the spatial integral representing intrinsic absorption corresponds exactly to:

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
the classical geometrical spreading factor :math:`L` relates the solid angle
of the ray tube at the source to its cross-sectional area at the receiver
(:footcite:t:`Cerveny2001`, :footcite:t:`AkiRichards2002`):

.. math::
    L = \sqrt{\frac{X \cdot \cos\theta_s \cdot \cos\theta_r}{p}
                            \left| \frac{\partial X}{\partial p} \right|}

where :math:`\theta_s, \theta_r` are the ray angles at source and
receiver. 
Equation above defines the **relative geometrical spreading** (see :footcite:t:`Cerveny2001`, Eq. 4.10.22), which measures the ray-tube geometrical divergence strictly from the ray curvature, without the source-point velocity multiplier :math:`1/v_r`. 

The derivative :math:`\partial X / \partial p` is computed
analytically via the chain rule:

.. math::
   \frac{\partial X}{\partial p}
   = \frac{\mathrm{d}X}{\mathrm{d}q} \cdot \frac{\mathrm{d}q}{\mathrm{d}p},
   \qquad
   \frac{\mathrm{d}q}{\mathrm{d}p}
   = \frac{v_{\max}}{(1 - p^2\,v_{\max}^2)^{3/2}}

----

Reflection and transmission coefficients
========================================

In layered media, wave amplitudes are modified at every crossed interface.
LayTracer supports two methods for computing interface coefficients: normal-incidence (impedance-only) approximation and angle-dependent P-SV formulation (Zoeppritz). Normal-incidence is there only for comparison purposes, as it is not physically accurate. The default method is angle-dependent P-SV formulation.

Normal-incidence (impedance-only) approximation
------------------------------------------------

For a wave crossing an interface from medium 1 to medium 2 at normal
incidence,

.. math::
   T = \frac{2 Z_1}{Z_1 + Z_2},
   \qquad Z_i = \rho_i v_i,

where :math:`Z_i` is the acoustic impedance and :math:`v_i` is the wave speed
of the considered mode (P or S).

Angle-dependent P-SV formulation (welded solid-solid interface)
---------------------------------------------------------------

For horizontal slowness (ray parameter) :math:`p`, the vertical slownesses are

.. math::
   \eta_{\alpha i} = \sqrt{\frac{1}{v_{Pi}^2} - p^2},
   \qquad
   \eta_{\beta i} = \sqrt{\frac{1}{v_{Si}^2} - p^2},

and the auxiliary quantities

.. math::
   a = \rho_2\!\left(1-2v_{S2}^2p^2\right)-\rho_1\!\left(1-2v_{S1}^2p^2\right),

.. math::
   b = \rho_2\!\left(1-2v_{S2}^2p^2\right)+2\rho_1v_{S1}^2p^2,

.. math::
   c = \rho_1\!\left(1-2v_{S1}^2p^2\right)+2\rho_2v_{S2}^2p^2,

.. math::
   d = 2\left(\rho_2v_{S2}^2-\rho_1v_{S1}^2\right).

Define the cosine-dependent intermediate terms

.. math::
   E=b\,\eta_{\alpha1}+c\,\eta_{\alpha2},
   \quad
   F=b\,\eta_{\beta1}+c\,\eta_{\beta2},
   \quad
   G=a-d\,\eta_{\alpha1}\eta_{\beta2},
   \quad
   H=a-d\,\eta_{\alpha2}\eta_{\beta1},

and the system determinant

.. math::
   D = EF + GH\,p^2.

The complete :math:`4\times 4` scattering matrix
(:footcite:t:`AkiRichards2002`, Eqs. 5.38–5.40) is computed by LayTracer.
The eight independent P-SV coefficients are listed below.

**Incident P-wave** — reflection and transmission:

.. math::
   R_{PP} = \frac{\left(b\,\eta_{\alpha1}-c\,\eta_{\alpha2}\right)F
             - \left(a+d\,\eta_{\alpha1}\eta_{\beta2}\right)H\,p^2}{D},

.. math::
   R_{PS} = -\frac{2\,\eta_{\alpha1}\left(ab+cd\,\eta_{\alpha2}\eta_{\beta2}\right)
             p\,(v_{P1}/v_{S1})}{D},

.. math::
   T_{PP} = \frac{2\rho_1\,\eta_{\alpha1}\,F\,(v_{P1}/v_{P2})}{D},

.. math::
   T_{PS} = \frac{2\rho_1\,\eta_{\alpha1}\,H\,p\,(v_{P1}/v_{S2})}{D}.

**Incident SV-wave** — reflection and transmission:

.. math::
   R_{SP} = -\frac{2\,\eta_{\beta1}\left(ab+cd\,\eta_{\alpha2}\eta_{\beta2}\right)
             p\,(v_{S1}/v_{P1})}{D},

.. math::
   R_{SS} = -\frac{\left(b\,\eta_{\beta1}-c\,\eta_{\beta2}\right)E
             - \left(a+d\,\eta_{\alpha2}\eta_{\beta1}\right)G\,p^2}{D},

.. math::
   T_{SP} = -\frac{2\rho_1\,\eta_{\beta1}\,G\,p\,(v_{S1}/v_{P2})}{D},

.. math::
   T_{SS} = \frac{2\rho_1\,\eta_{\beta1}\,E\,(v_{S1}/v_{S2})}{D}.

For references and details on the derivation of these formulas, see
:footcite:t:`LayWallace1995` (Table 3.1, note the sign error in the second
term of :math:`b`) and :footcite:t:`AkiRichards2002` (Equations 5.38–5.40).

The angle-dependent formulation is used by default. It reduces to the
normal-incidence expression for :math:`p=0`.
For post-critical incidence the coefficients may become complex; for
amplitude modelling the software uses :math:`|T_l|`.

Critical angles
---------------

A **critical angle** occurs when the transmitted wave in a faster medium
becomes evanescent.  For an incident P-wave crossing into a layer where
:math:`v_{P2} > v_{P1}`, the P-critical angle is

.. math::
   \theta_c^{P} = \arcsin\!\left(\frac{v_{P1}}{v_{P2}}\right).

Similarly, when :math:`v_{S2} > v_{P1}` an S-to-P critical angle exists:

.. math::
   \theta_c^{S} = \arcsin\!\left(\frac{v_{P1}}{v_{S2}}\right).

For an incident SV-wave the same logic applies with :math:`v_{S1}` in the
numerator.  Beyond the critical angle the corresponding vertical slowness
becomes imaginary, the coefficient becomes complex, and total reflection
occurs for that mode.

Brewster angles
---------------

By analogy with optics, a **Brewster angle** is an incidence angle at which a
reflection or transmission coefficient passes through zero or a deep minimum.
In electromagnetic theory, Brewster's angle is the incidence angle at which
:math:`R_p = 0` for p-polarised light at a dielectric interface.  In
elastodynamics, the same phenomenon occurs: certain combinations of elastic
parameters produce incidence angles where one of the P-SV scattering
coefficients vanishes.

Unlike critical angles, which depend only on velocity ratios, Brewster angles
depend on *all six* elastic parameters (:math:`v_{P1}`, :math:`v_{S1}`,
:math:`\rho_1`, :math:`v_{P2}`, :math:`v_{S2}`, :math:`\rho_2`).
Physically, they arise from destructive interference between the P and SV
displacement potentials at the welded interface: the two potential
contributions to a particular scattered mode cancel exactly, driving that
coefficient to zero.

For example, the reflected P coefficient :math:`R_{PP}` may vanish at an
angle well below the critical angle.  At this Brewster angle the incident
energy is partitioned entirely into the transmitted P-wave and the
mode-converted waves, with no same-mode reflection.

LayTracer provides the function :func:`~laytracer.amplitude.find_brewster_angles`
which numerically detects these minima in the computed coefficient curves by
searching for local minima of :math:`|C(\theta)|` whose value falls below a
user-specified threshold.

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
- **Geometrical spreading** — the formula incorporates the
  epicentral distance :math:`X` to capture the 3-D cylindrical
  divergence of the ray-tube out of the incidence plane
  (:footcite:t:`Cerveny2001`, §4.10).
- **Transmission coefficients** — depend on ray parameter and layer
  impedances, not on azimuth.

Thus, the layered-media solver need only operate in the 2-D ray plane;
the full 3-D solution is recovered by geometry alone.

----

References
==========

.. footbibliography::
