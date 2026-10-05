.. _contributing:

============
Contributing
============

Contributions, bug reports and suggestions are welcome.

Reporting a problem
===================

Please open an issue at
`github.com/danikiev/LayTracer/issues <https://github.com/danikiev/LayTracer/issues>`_.
A useful report states the LayTracer version, the Python version, what you expected,
what happened instead, and a small script that reproduces the behaviour, including the
velocity model and the source and receiver positions.

For a suspected error in a traveltime or an amplitude attribute, the most useful report
compares the result against a closed-form expectation, such as ``dist / v`` in a
homogeneous medium, because that makes the discrepancy measurable.

Development setup
=================

.. code-block:: bash

   git clone https://github.com/danikiev/LayTracer.git
   cd LayTracer
   conda env create -f environment.yml
   conda activate laytracer
   pip install -e .

The ``laytracer`` environment already includes the test and documentation tools.

Branches and releases
=====================

Development happens on ``dev``, and releases are made from ``main``. Release tags follow
semantic versioning as ``vMAJOR.MINOR.PATCH`` and must point at a commit contained in
``main``; the version itself is derived from the tag by ``setuptools_scm``. The full
procedure, including the TestPyPI and PyPI uploads, is described in ``RELEASE.md``.

Tests
=====

Run the suite from the repository root:

.. code-block:: bash

   python -m pytest pytests/

Continuous integration runs the same suite on Python 3.8 to 3.12, validates the Zenodo
metadata, and builds this documentation.

Please add tests with any change to the solver or to the amplitude attributes. The
convention in this project is to check results against closed forms, such as
``dist / v`` in a homogeneous medium, the sum of ``h / v`` over the layers for a vertical
ray, a transmission coefficient of one between identical media, and ``t* = tt / Q`` for
uniform attenuation, and to check reciprocity: the traveltime and ``t*`` from A to B
must equal those from B to A.

Documentation
=============

Build the documentation with the helper scripts, which then serve it locally:

.. code-block:: bash

   ./build-docs.sh          # Linux, macOS;  build-docs.bat on Windows
   ./build-docs.sh -pdf     # also builds the PDF edition

Without the scripts the equivalent is ``make -C docs html``, or ``docs\make.bat html`` on
Windows.

Docstrings follow the `numpydoc <https://numpydoc.readthedocs.io>`_ style. Formulas
reference the equation numbers of the Fang and Chen (2019) paper that LayTracer
implements, equations use ``.. math::``, and literature is cited with ``:cite:t:`` or
``:cite:p:`` against ``docs/source/references.bib``.

Changelog
=========

``CHANGELOG.md`` follows `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_. Add
your entry under ``[Unreleased]`` in the appropriate section, referencing the issue or
pull request number. The documentation page is generated from this file, so there is
nothing else to update.
