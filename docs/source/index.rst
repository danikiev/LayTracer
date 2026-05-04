.. _overview:

========
Overview
========

.. only:: html

   .. image:: _static/laytracer-logo-full.svg
      :alt: LayTracer logo
      :align: center
      :width: 520px

**LayTracer** is an open-source Python package for computing ray paths, travel times, and amplitude attributes in horizontally layered (1D) velocity models with constant layer velocities. It is based on the dimensionless ray parameter method of :cite:t:`FangChen2019`, achieving rapid convergence.

**Current Version:** |release| (:ref:`changelog`)

.. only:: html
   
   **Latest DOI:** |DOI| (:ref:`citing`)

   .. |DOI| image:: https://zenodo.org/badge/1160026484.svg
         :target: https://zenodo.org/badge/latestdoi/1160026484

.. only:: latex

   **This documentation online:** `danikiev.github.io/LayTracer <https://danikiev.github.io/LayTracer>`_

   **DOI:** `10.5281/zenodo.18850919 <https://doi.org/10.5281/zenodo.18850919>`__ (:ref:`citing`)

----

**Features:**

* Fast two-point ray tracing via dimensionless ray parameter method
* Second-order Newton iteration for rapid convergence
* Refraction and reflection modes
* Inline computation of travel time, attenuation operator :math:`t^*`, geometrical spreading, and reflection/transmission coefficients
* Efficient parallel computations via `Joblib <https://joblib.readthedocs.io/>`_
* Standalone `Matplotlib <https://matplotlib.org/>`_ / `Plotly <https://plotly.com/>`_ visualisation
* Comprehensive `Sphinx <https://www.sphinx-doc.org/>`_ documentation with extensive theory available at `danikiev.github.io/LayTracer <https://danikiev.github.io/LayTracer>`_

.. only:: html

   ----

   **Quick Links:**

   .. grid:: 1 2 2 2
      :gutter: 3
        
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

      .. grid-item-card::
         :link: citing
         :link-type: ref
         :link-alt: citing

         :fas:`quote-right;pst-color-primary` **Citing**
         ^^^
         Find citation information and BibTeX entries for referencing LayTracer in your research.

      .. grid-item-card::
         :link: credits
         :link-type: ref
         :link-alt: credits

         :fas:`users;pst-color-primary` **Credits**
         ^^^
         Acknowledge authors, contributors and used libraries in the development of LayTracer.

      .. grid-item-card::
         :link: changelog
         :link-type: ref
         :link-alt: changelog

         :fas:`clock-rotate-left;pst-color-primary` **Changelog**
         ^^^
         Review release notes and notable changes across versions.

      .. grid-item-card::
         :link: _static/laytracer.pdf     
         :link-alt: pdf

         :fas:`file-pdf;pst-color-primary` **Download as PDF**
         ^^^
         Download this documentation as a standalone :fas:`file-pdf` PDF file.

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
   citing
   credits
   changelog
