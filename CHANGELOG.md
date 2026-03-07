# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `normalize_rt_coefficient()` function in `amplitude.py` implementing
  Červený (2001) Eq. 5.3.10 energy-flux normalization of R/T coefficients ([#1])
- `transcoef_method="normalized"` option in `trace_rays` and `solve` for
  energy-flux-normalized transmission coefficient products ([#1])
- new tests: `test_transmission_normalized`, `test_normalized_vertical_ray` ([#1])
- methodology docs: section on energy-flux-normalized coefficients ([#1])

### Changed

- `transcoef_method` values renamed: `"angle"` → `"standard"`,
  `"angle_normalized"` → `"normalized"` ([#1])
- default `transcoef_method` is now `"standard"` (was `"angle"`) ([#1])
- updated documentation to reflect new method names and default ([#1])
- updated example 5 to use new method names and default ([#1])
- updated example 4 to compare standard vs normalized transmission coefficients ([#1])
- updated example 3 to show comparison of standard vs normalized transmission and reflection coefficients ([#1])

### Remov
- `"normal"` (impedance-only) transmission coefficient method — only
  `"standard"` and `"normalized"` are supported ([#1])
- normal-incidence section removed from methodology docs ([#1])

[#1]: https://github.com/anikiev-icd/LayTracer/issues/1

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
