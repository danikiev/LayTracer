# Changelog

All notable changes to Laytracer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- LayTracer logo and branding assets, including full, medium, icon, circular icon, and PDF-compatible logo variants (#12)
- deterministic logo generation script under `branding/logo/` with vendored Poppins font assets and staleness checking (#12)
- logo integration in the repository README, HTML documentation overview, Sphinx header, favicon, and LaTeX/PDF title page (#12)
- branding documentation describing the logo design concept, color palette, generated outputs, usage locations, and regeneration commands (#12)

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
