.. _getting_started:

===============
Getting Started
===============

This chapter provides everything you need to begin using LayTracer. 
It includes :ref:`installation instructions<installation>` including :ref:`dependency management<dependencies>` and general description of the :ref:`package design<package_design>`.

----

.. _installation:

Installation
============

.. _requirements:

Requirements
------------

Installation requires `Conda <https://conda.io>`_ package manager. 
We recommend using the `miniforge <https://github.com/conda-forge/miniforge>`_ implementation.

.. _dependencies:

Dependencies
------------

LayTracer leverages numerous Python packages. 
Below are some of the key dependencies:

* `Python <https://www.python.org>`_ 3.8 to 3.12
* `NumPy <https://www.numpy.org>`_ 1.20 to 1.26
* `SciPy <https://www.scipy.org>`_
* `Pandas <https://pandas.pydata.org/>`_
* `Numba <https://numba.pydata.org/>`_
* `Matplotlib <https://matplotlib.org/>`_
* `Plotly <https://plotly.com/python/>`_
* `Sphinx <https://www.sphinx-doc.org>`_ (used to create this documentation)

The authors are incredibly grateful to the developers of these packages.

.. _install_using_conda:

Install using conda
-------------------

The best way to install is by creating a new conda environment with all required packages:

.. code-block:: bash

    conda env create -f environment.yml

**Note:** to speed up creation of the environment, use `mamba` instead of `conda`, which is a faster alternative.

Then activate the newly created environment:

.. code-block:: bash

    conda activate laytracer

Finally, install the package:

.. code-block:: bash

    pip install -e .

.. _install_using_script:

Install using script
--------------------

For quick installation, you can use the specially designed installation scripts which implement all of the above mentioned steps.

On Windows, in miniforge prompt run:

.. code-block:: batch

    install.bat

.. _uninstall:

Uninstall
---------

If you need to add/change packages, deactivate the environment first:

.. code-block:: bash

    conda deactivate

Then remove the appropriate environment:

.. code-block:: bash

    conda remove -n laytracer --all

----

.. _package_design:

Package Design
==============

LayTracer is structured as a :fab:`python;pst-color-primary` Python package.

