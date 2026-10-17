# Changelog

All notable changes to Laytracer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- the documentation of a release reaches GitHub Pages. After v0.5.0 the site kept showing ``0.4.1.dev38+gf9703a9be``, because Pages ignores a deployment made from a tag when ``main`` has already deployed the same commit ([actions/deploy-pages#383](https://github.com/actions/deploy-pages/issues/383)). Pushing a ``v*`` tag now starts a ``docs.yml`` run on ``main`` that builds and deploys the tag, only runs on ``main`` deploy, and each deployment checks that the live site shows the version it deployed
- the quadratic Newton step in q no longer stalls at large offsets. Its small root came from the textbook formula, which cancels to a zero step once ``4|AC|`` drops below machine precision relative to ``B**2``; the bracketed fallback then finished the solve, and the diagnostics reported ``q_brentq`` for rays that are not ill-conditioned. In a seeded benchmark this affected 5% of rays at the default tolerance and 14% at ``tol=1e-6``, almost all beyond ten times the source depth. The roots are now evaluated in cancellation-free form, so these rays converge in the quadratic step; travel times at offsets below ten times the source depth change by less than 1e-7 s (#17)

## [v0.5.0] - 2026-10-12

### Added

- ordered ``Interaction`` and ``RayItinerary`` objects for prescribed direct, reflected, transmitted, converted, and multiple phase paths
- opt-in per-ray ``SolveDiagnostics`` with accepted solve route, iteration counts, transformed and physical ray parameters, independently recomputed endpoint residual, conditioning, and criticality margin
- sparse fixed-topology analytic derivatives of traveltime and physical ray parameter with respect to layer ``Vp``/``Vs``, interface depths, and source and receiver coordinates
- reusable linearized-ray, point-cloud anchor-selection, and topology-safe dense traveltime approximation APIs
- opt-in second-order endpoint traveltime prediction, including the zero-offset curvature limit and exact direct same-layer paths
- a visual sensitivity example showing scaled traveltime and ray-parameter effects, local model prediction, dense endpoint prediction, and a vertical-well microseismic traveltime table approximated from sparse source anchors
- optional ``complex_coefficient_product`` output alongside the established real coefficient-magnitude product
- LayTracer logo and branding assets, including full, medium, icon, circular icon, and PDF-compatible logo variants (#12)
- deterministic logo generation script under `branding/logo/` with vendored Poppins font assets and staleness checking (#12)
- logo integration in the repository README, HTML documentation overview, Sphinx header, favicon, and LaTeX/PDF title page (#12)
- branding documentation describing the logo design concept, color palette, generated outputs, usage locations, and regeneration commands (#12)
- a Contributing page in the documentation, covering issue reports, the development setup, branches and releases, tests, the documentation build, and the changelog; it replaces the short contributing note in Credits
- a section in the README and in Getting Started on limiting NumPy's BLAS threads with ``OPENBLAS_NUM_THREADS``, because on Windows some OpenBLAS builds commit a buffer for every thread when NumPy is imported (#15)

### Changed

- the transformed-parameter solver now uses a checked bracketed fallback and accepts a result only after independently verifying the endpoint offset
- density and quality-factor columns are validated only when a requested output depends on them
- legacy ``reflection`` and ``refraction`` tuples compile through the same fixed-itinerary representation; ``refraction`` continues to mean prescribed transmission or mode conversion rather than a head wave
- new scientific outputs remain additive and opt-in; the 0.4 public ``trace_rays`` argument order, result flattening, phase dispatch, and default
  allocations are unchanged
- exact-endpoint anchor fitting and per-pair prediction bookkeeping are streamlined for dense traveltime tables
- ``rays_per_chunk`` now caps the rays per parallel task rather than per chunk. Left as ``None``, the task size follows from the number of rays and workers and from the output size per ray, measured on a probe of up to 64 rays; ``trace_rays`` issues a ``RuntimeWarning`` when the estimated output exceeds available memory (#15)
- ``n_jobs=1`` above ``sequential_limit`` uses the same bounded-memory path, in-process (#15)
- verbose tracing prints the task plan and the estimated output size, then progress in 10% steps, instead of one line per chunk (#15)
- ``joblib>=1.4`` is required, for results streamed back as tasks finish (#15)
- loky workers trace with one BLAS/OpenMP thread each (joblib's ``inner_max_num_threads=1``), because tracing makes no BLAS calls. On Windows, OpenBLAS commits a buffer for every thread when NumPy is imported, about 135 MB per thread with conda-forge builds, so the default of CPUs divided by workers could cost a worker over 1 GB (#15)
- LayTracer requires NumPy 2 (``numpy>=2``) and Python 3.9 or newer, without the previous upper bound of 3.13; continuous integration tests Python 3.9 to 3.14
- dependencies are declared only in ``pyproject.toml``, as in PyNetDesign: a new ``dev`` extra adds pytest and setuptools_scm next to the existing ``docs`` extra, and ``install.bat`` and ``install.sh`` install ``".[dev]"``. Installs therefore take NumPy from its PyPI wheels, which bundle OpenBLAS
- the documentation's project pages follow the main chapters in the order Citing, Contributing, Changelog, Credits, as do the panels on the overview, where the PDF download now spans the full width
- chapters of the PDF start on the next page rather than the next right-hand one, so the PDF has no blank pages
- the documentation workflow builds with Python 3.12, which Sphinx 9.1 requires
- the changelog page shows version headings without brackets, each followed by a link to its changes on GitHub, taken from the link definitions now at the end of ``CHANGELOG.md``; code written between double backticks there renders as code instead of showing stray backticks

### Removed

- support for Python 3.8 and for NumPy 1
- the automatic ``rays_per_chunk`` value, 50% of available RAM divided by the number of workers at an assumed 64 bytes per ray, and its ``Auto-detected rays_per_chunk`` message. No parameter was removed: ``rays_per_chunk`` remains and now caps the rays per task, sized as described under Changed (#15)
- the package list in ``environment.yml``, which now provides only Python and ``pip``; the runtime, test and documentation packages are installed from ``pyproject.toml``
- the ``release.yml`` workflow that published packages when a tag was pushed; packages are now published only from a published GitHub Release, after the tag and the Zenodo metadata are validated

### Fixed

- parallel ``trace_rays`` no longer runs out of memory on large grids, where it failed with ``MemoryError`` in a loky worker sending its results back. Tasks now return NumPy columns that are copied into the output as soon as each task finishes, instead of one Python tuple per ray held for a whole chunk, first in the workers and then in the main process. For travel times of two phases the 0.4 chunk size assumed 64 bytes per ray, while the tuples, their pickling and the indices needed close to 1 kB per ray; memory use is now the output plus a few tasks per worker (#15)
- reflected and multiple-ray spreading now uses the actual source- and receiver-side phase velocities (#13)
- coefficient products retain complex sign and phase when explicitly requested while ``trans_product`` remains a real, nonnegative magnitude for
  existing workflows
- 2-D ray plotting uses the public pyplot colormap accessor and remains compatible with Matplotlib 3.11, where ``matplotlib.cm.get_cmap`` was removed
- the PDF builds with the June 2026 LaTeX release, where tables in the default ``colorrows`` style of Sphinx 9.1.0 exhausted the TeX input stack (sphinx-doc/sphinx#14465)
- citations in the PDF link to its bibliography; they pointed at the reference lists that only the HTML pages contain
- ``build-docs.bat`` detects failed steps inside its blocks, and both build scripts stop at a LaTeX error instead of waiting at the TeX prompt
- ``make cleanall`` removes the generated example gallery in ``docs/source/examples``
- a push to ``dev`` no longer cancels a running documentation build of ``main`` or of a release tag, and with it the Pages deployment

## [v0.4.0] - 2026-05-02

### Added

- `trace_rays(..., source_phase=...)` now accepts a list of phases, for example `["P", "SH", "SV"]`, and returns a `dict[str, TraceResult]` for multi-phase requests (#7)
- explicit SH-wave ray tracing support, including SH-SH reflection/transmission coefficients for welded isotropic interfaces (#7)
- `TraceResult.source_phase` to record the canonical traced phase (#7)
- regression tests for multi-phase API behavior, S/SV aliasing, SV/SH shared kinematics, SH energy conservation, SH post-critical behavior, and upward-path SV/SH equality for `tstar` and spreading (#7)
- coefficient-panel styling for complex-valued coefficient segments and truly evanescent outgoing branches, with regression tests for the plot helper and SH reflected-coefficient null
- optional `layer_colors` and `ray_linewidth` arguments for `laytracer.plot.rays_2d()`, with regression tests for custom layer palettes, ray widths, and unpadded x limits

### Changed

- canonicalized phase handling so legacy `S` maps to `SV`, while `P`, `SV`, and `SH` are available as explicit source phases (#7)
- multi-phase tracing reuses kinematic solves across phases when possible; SV and SH share ray paths, travel times, ray parameters, `tstar`, and geometrical spreading, while retaining phase-specific transmission/reflection products (#7)
- example 03 now includes SH reflection/transmission cases, distinguishes complex coefficient phase shifts from evanescent outgoing branches, and annotates the SH oblique impedance-match null (#7)
- clarified the methodology narrative for SH in 3-D layered media, distinguishing 2-D incidence-plane ray geometry from transverse SH polarization (#7)
- expanded example 04 with an SV/SH comparison showing shared S-wave kinematics and phase-specific interface coefficient products (#7)
- example 03 ray-path diagrams now use a seven-color ColorBrewer Accent palette, explicit incident-ray paths, solid ray lines, and clearer layer/ray contrast
- cleaned up and refactored example 03
- API and methodology documentation now describe SH support, multi-phase tracing, and the decoupled SH behavior in isotropic 1-D media (#7)

### Fixed

- Removed unnecessary `plt.show()` call from 3D ray plot in example 01 to enable 3D plot in docs pages

## [v0.3.1] - 2026-04-20

### Added

- `critical_angle()` and `find_critical_angles()` helpers in `laytracer.amplitude`
  for reusable critical-angle detection
- `laytracer.plot.coefficient_panels()` for reusable multi-panel plotting of
  reflection/transmission coefficient curves
- new regression tests for the critical-angle helpers and coefficient-panel
  plotting helper
- citing docs section with updated DOI and citation info
- credits docs section with author and contributor acknowledgements and library credits

### Changed

- refactored example 3 to use the new critical-angle and coefficient-panel
  helpers instead of open-coded calculations and repeated subplot logic
- updated the API reference to include the new amplitude and plotting helpers
- docs index page revised

## [v0.3.0] - 2026-03-14

### Added

- add explicit `requested={...}` output selection to `trace_rays()` and `solve()`, replacing the coarse amplitude switch (#3)

### Changed

- only build and return ray paths, ray parameters, and path-dependent scalar outputs when explicitly requested (#3)

### Fixed

- fix degenerate direct-ray amplitude outputs for zero-offset vertical rays and
  exact same-point rays (#2)
- make amplitude result packing robust to mixed `None` and finite values (#2)
- add regression tests for degenerate direct-ray amplitude cases (#2)

## [v0.2.1] - 2026-03-07

### Added

- `normalize_rt_coefficient()` function in `amplitude.py` implementing
  Červený (2001) Eq. 5.3.10 energy-flux normalization of R/T coefficients (#1)
- `transcoef_method="normalized"` option in `trace_rays` and `solve` for
  energy-flux-normalized transmission coefficient products (#1)
- new tests: `test_transmission_normalized`, `test_normalized_vertical_ray` (#1)
- methodology docs: section on energy-flux-normalized coefficients (#1)

### Changed

- `transcoef_method` values renamed: `"angle"` → `"standard"`,
  `"angle_normalized"` → `"normalized"` (#1)
- default `transcoef_method` is now `"standard"` (was `"angle"`) (#1)
- updated documentation to reflect new method names and default (#1)
- updated example 5 to use new method names and default (#1)
- updated example 4 to compare standard vs normalized transmission coefficients (#1)
- updated example 3 to show comparison of standard vs normalized transmission and reflection coefficients (#1)

### Removed

- `"normal"` (impedance-only) transmission coefficient method — only
  `"standard"` and `"normalized"` are supported (#1)
- normal-incidence section removed from methodology docs (#1)

## [v0.2.0] - 2026-03-04

### Added

- `ModelArrays` dataclass for pre-extracted NumPy arrays, avoiding repeated
  DataFrame column extraction during parallel tracing
- fast path in `_trace_one` for direct waves (no reflections/refractions)
- memory-aware `rays_per_chunk` auto-sizing and progress reporting with ETA
  in `trace_rays`

### Changed

- `build_layer_stack` now accepts both `pd.DataFrame` and `ModelArrays`
  (unified from the former `build_layer_stack` / `build_layer_stack_fast` pair)
- batched parallel dispatch: rays are grouped into ~n_workers batches with
  lightweight NumPy-only serialisation instead of per-ray DataFrame pickling
- updated documentation index page

### Removed

- dead first-pass loop in `_trace_one` that built unused variables

### Fixed

- handle degenerate case in `_trace_one` function to return minimal result
- fix NaN results for same-depth source–receiver rays (e.g. stations and grid
  points both at z = 0): zero-thickness layer stack is now handled as a
  horizontal straight-line ray with correct travel time, geometrical spreading,
  attenuation t*, and transmission product instead of returning NaN

## [v0.1.0] - 2026-03-03

### Added

- Files for initial release
- This changelog
- GitHub Actions CI for pytest, docs build, and release automation

[Unreleased]: https://github.com/danikiev/LayTracer/compare/v0.5.0...HEAD
[v0.5.0]: https://github.com/danikiev/LayTracer/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/danikiev/LayTracer/compare/v0.3.1...v0.4.0
[v0.3.1]: https://github.com/danikiev/LayTracer/compare/v0.3.0...v0.3.1
[v0.3.0]: https://github.com/danikiev/LayTracer/compare/v0.2.1...v0.3.0
[v0.2.1]: https://github.com/danikiev/LayTracer/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/danikiev/LayTracer/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/danikiev/LayTracer/releases/tag/v0.1.0
