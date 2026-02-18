.. LayTracer documentation master file

========
Overview
========

**LayTracer** is a Python implementation of fast two-point seismic ray
tracing in 1-D layered media with constant layer velocity, based on
the dimensionless ray parameter method of
:cite:t:`FangChen2019`.

Features
========

* Second-order Newton iteration for rapid convergence (2–3 iterations)
* Inline computation of travel time, :math:`t^*`, geometrical spreading,
  and transmission coefficients
* Parallel computation via ``loky`` backend (joblib)
* Standalone matplotlib/plotly visualisation
* Comprehensive Sphinx documentation with full mathematical derivations


----

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Contents

   self   
   getting_started
   methodology
   examples/index
   api/index

.. only:: html

.. grid:: 1 2 4 4
   
   .. grid-item-card::
      :link: getting_started
      :link-type: ref
      :link-alt: getting started
      
      :fas:`play;pst-color-primary` **Getting Started**
      ^^^
      Install LayTracer, set up dependencies, and prepare input files.

   .. grid-item-card:: 
      :link: methodology
      :link-type: ref
      :link-alt: methodology

      :fas:`book;pst-color-primary` **Methodology**
      ^^^
      Understand core algorithms for seismic ray tracing.

   .. grid-item-card:: 
      :link: examples
      :link-type: ref
      :link-alt: examples

      :fas:`lightbulb;pst-color-primary` **Examples**
      ^^^
      Explore practical use cases of LayTracer in action.

   .. grid-item-card:: 
      :link: api
      :link-type: ref
      :link-alt: api

      :fas:`code;pst-color-primary` **API Reference**
      ^^^
      Access detailed API documentation for all modules.

.. grid:: 1

   .. grid-item-card::
      :link: _static/laytracer.pdf     
      :link-alt: pdf

      :fas:`file-pdf;pst-color-primary` **Download as PDF**
      ^^^
      Download this documentation as a standalone :fas:`file-pdf` PDF file.

----

.. bibliography::
    :style: unsrt
    :filter: docname in docnames
