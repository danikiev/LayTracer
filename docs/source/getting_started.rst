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

LayTracer can be installed either with `Conda <https://conda.io>`_ or with standard Python ``pip``.
For a reproducible full environment, we recommend using the `miniforge <https://github.com/conda-forge/miniforge>`_ Conda implementation.

.. _dependencies:

Dependencies
------------

LayTracer leverages numerous Python packages. 
Below are some of the key dependencies:

* `Python <https://www.python.org>`_ 3.8 to 3.12
* `NumPy <https://www.numpy.org>`_ 1.20 to 1.26
* `SciPy <https://www.scipy.org>`_
* `Pandas <https://pandas.pydata.org/>`_
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

.. _install_using_pip:

Install using pip only
----------------------

If you prefer not to use Conda, you can install LayTracer into a standard virtual environment.

On Linux / macOS:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -e .

On Windows (PowerShell):

.. code-block:: powershell

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -e .

The pip-only approach is appropriate for package usage. For a pre-configured environment including documentation tooling, prefer the Conda workflow.

.. _install_using_script:

Install using script
--------------------

For quick installation, you can use the specially designed installation scripts which implement all of the above mentioned steps.

On Windows, in miniforge prompt run:

.. code-block:: batch

    install.bat

On Linux / macOS, from the repository root run:

.. code-block:: bash

    chmod +x install.sh
    ./install.sh

.. _build_docs_using_script:

Build documentation using script
--------------------------------

To build and serve the documentation quickly, use the platform-specific scripts.

On Windows:

.. code-block:: batch

    build-docs.bat

Build HTML + PDF on Windows:

.. code-block:: batch

    build-docs.bat -pdf

On Linux / macOS:

.. code-block:: bash

    chmod +x build-docs.sh
    ./build-docs.sh

Build HTML + PDF on Linux / macOS:

.. code-block:: bash

    chmod +x build-docs.sh
    ./build-docs.sh -pdf

.. _build_docs_using_make:

Build documentation using make
------------------------------

If you prefer explicit Sphinx/Make commands, you can build the docs directly from the ``docs`` folder.

On Linux / macOS:

.. code-block:: bash

    cd docs
    make html

Build HTML + PDF on Linux / macOS:

.. code-block:: bash

    cd docs
    make html
    make latexpdf

On Windows:

.. code-block:: batch

    cd docs
    make.bat html

Build HTML + PDF on Windows:

.. code-block:: batch

    cd docs
    make.bat html
    make.bat latexpdf

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

LayTracer is structured as a :fab:`python;pst-color-primary` Python package with a layered, modular architecture designed for numerical reliability, API clarity, and maintainability.

High-level architecture
-----------------------

The implementation is separated into focused modules with clear responsibilities:

* ``laytracer.model``

    * Defines model-level data containers (for example, ``LayerStack``).
    * Converts tabular velocity/depth input into solver-ready layer stacks.

* ``laytracer.solver``

    * Implements the core two-point ray tracing algorithm (dimensionless-parameter Newton solver).
    * Computes per-ray kinematics such as travel time and ray geometry.

* ``laytracer.amplitude``

    * Implements amplitude-related physics (transmission/reflection coefficients, Brewster-angle analysis).
    * Provides reusable low-level functions independent of plotting/UI concerns.

* ``laytracer.api``

    * Exposes high-level user workflows (for example, multi source-receiver tracing via ``trace_rays``).
    * Handles batching, parallel execution, and result aggregation into user-facing containers.

* ``laytracer.plot``

    * Provides visualization utilities (2-D profiles and 3-D interactive rendering).
    * Keeps visualization optional and decoupled from numerical core logic.

Public API surface
------------------

The package root (``laytracer.__init__``) re-exports the primary classes/functions so typical user workflows stay concise while internal module boundaries remain explicit.

Design principles
-----------------

LayTracer follows several engineering principles:

* **Separation of concerns**: numerical solvers, physical coefficients, API orchestration, and plotting are isolated.
* **Composability**: low-level building blocks can be used independently in custom workflows.
* **Performance-aware implementation**: vectorized NumPy operations and optional parallel execution for survey-scale runs.
* **Reproducibility**: environment-driven dependency management and deterministic, test-backed numerical behavior.
* **Extensibility**: new physical attributes or workflow wrappers can be added without rewriting the solver core.

Typical execution flow
----------------------

1. Build a layered model from input data (``build_layer_stack`` or high-level API input).
2. Solve one or many source-receiver ray paths (``solve`` / ``trace_rays``).
3. Optionally compute attenuation/spreading/transmission attributes.
4. Visualize outputs using ``laytracer.plot`` helpers.

