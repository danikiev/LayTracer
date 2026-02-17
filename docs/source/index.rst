.. LayTracer documentation master file

========
LayTracer
========

**LayTracer** is a Python implementation of fast two-point seismic ray
tracing in 1-D layered media with constant layer velocity, based on
the dimensionless ray parameter method of
:footcite:t:`FangChen2019`.

Features
--------

* Second-order Newton iteration for rapid convergence (2–3 iterations)
* Inline computation of travel time, :math:`t^*`, geometrical spreading,
  and transmission coefficients
* Parallel computation via ``loky`` backend (joblib)
* Standalone matplotlib/plotly visualisation
* Comprehensive Sphinx documentation with full mathematical derivations

.. toctree::
   :maxdepth: 2
   :caption: Contents

   theory
   api/index
   examples/index

References
----------

.. footbibliography::
