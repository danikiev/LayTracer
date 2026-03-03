.. _overview:

========
Overview
========

**LayTracer** is an open-source Python package for computing ray paths, travel times, and amplitude attributes in horizontally layered (1D) velocity models with constant layer velocities. It is based on the dimensionless ray parameter method of :cite:t:`FangChen2019`, achieving rapid convergence.

**Version:** |release|

**Documentation:** `danikiev.github.io/LayTracer <https://danikiev.github.io/LayTracer>`_

----

**Features:**

* Fast two-point ray tracing via dimensionless ray parameter method
* Second-order Newton iteration for rapid convergence
* Refraction and reflection modes
* Inline computation of travel time, attenuation operator :math:`t^*`, geometrical spreading, and reflection/transmission coefficients
* Efficient parallel computations via `Joblib <https://joblib.readthedocs.io/>`_
* Standalone `Matplotlib <https://matplotlib.org/>`_ / `Plotly <https://plotly.com/>`_ visualisation
* Comprehensive `Sphinx <https://www.sphinx-doc.org/>`_ documentation with extensive theory available at `danikiev.github.io/LayTracer <https://danikiev.github.io/LayTracer>`_

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
