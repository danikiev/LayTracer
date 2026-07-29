# LayTracer

<p align="center">
  <img src="docs/source/_static/laytracer-logo-full.svg" alt="LayTracer logo" width="720">
</p>

[![DOI](https://zenodo.org/badge/1160026484.svg)](https://zenodo.org/badge/latestdoi/1160026484)
[![Pytest](https://github.com/danikiev/LayTracer/actions/workflows/pytest.yml/badge.svg)](https://github.com/danikiev/LayTracer/actions/workflows/pytest.yml)
[![Docs](https://github.com/danikiev/LayTracer/actions/workflows/docs.yml/badge.svg)](https://github.com/danikiev/LayTracer/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](https://github.com/danikiev/LayTracer/blob/main/LICENSE)

**Fast two-point seismic ray tracing in layered media.**

LayTracer is an open-source Python package for computing ray paths, travel times, and amplitude attributes in horizontally layered (1D) velocity models with constant layer velocities.
It also computes sparse fixed-topology derivatives of traveltime and ray parameter with respect to layer velocities, interface depths, and source/receiver coordinates for sensitivity analysis and inversion.
The ray tracing core algorithm is based on the dimensionless ray parameter method of [Fang & Chen (2019)](https://doi.org/10.1111/1365-2478.12799), achieving rapid convergence.

Documentation: [https://danikiev.github.io/LayTracer](https://danikiev.github.io/LayTracer/)

---

## ✨ Features

| Category | Capability |
| :------- | :--------- |
| **Ray tracing** | Second-order dimensionless-*q* iteration with a checked bracketed fallback and independently verified endpoint residual |
| **Phase itineraries** | Explicit prescribed direct, reflected, transmitted, converted, and multiple P–SV/SH paths |
| **Diagnostics** | Opt-in solve route, iterations, residual, conditioning, and criticality margin for every ray |
| **Sensitivities** | Sparse fixed-topology derivatives of traveltime and physical ray parameter with respect to layer velocities, interface depths, and endpoints |
| **Travel time** | Layer-by-layer travel time summation from the solved ray parameter |
| **Attenuation** | Intrinsic absorption operator *t\** from quality factors *Q* |
| **Relative amplitude attributes** | Geometrical spreading, attenuation, and angle-dependent P-SV/SH coefficient products; these are not complete synthetic amplitudes |
| **Brewster angles** | Automatic detection of Brewster-like zeros in R/T coefficient curves |
| **Parallel execution** | Multi-ray tracing with `joblib` / `loky` backend for large surveys |
| **Visualisation** | 2-D ray path plots (matplotlib) and interactive 3-D viewer (Plotly) |
| **Documentation** | Comprehensive [Sphinx docs](https://danikiev.github.io/LayTracer/) with extensive theory, gallery examples, and API reference |

---

## 📦 Installation

### Prerequisites

- Python 3.8–3.12 and `pip`
- [Conda](https://conda.io) package manager (recommended for full reproducible setup via [miniforge](https://github.com/conda-forge/miniforge))

### Install with conda

```bash
# 1. Create environment with all dependencies
conda env create -f environment.yml

# 2. Activate it
conda activate laytracer

# 3. Install LayTracer in editable mode
pip install -e .
```

### Install from PyPI (stable releases only)

```bash
python -m pip install --upgrade pip
pip install laytracer
```

### Check installed version

```bash
python -c "import laytracer; print(laytracer.__version__)"
```

Alternative:

```bash
python -m pip show laytracer
```

Use this mode when you want a published stable release. For development or latest unreleased changes, install from the repository with `pip install -e .`.

### Quick install (Windows)

```batch
install.bat
```

### Quick install (Linux / macOS)

```bash
chmod +x install.sh
./install.sh
```

### Install with pip only (no conda)

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

`pip`-only installation is suitable for running LayTracer. For a full pre-configured environment (including docs tooling), prefer the conda workflow.

### Dependencies

- Python ≥ 3.8, < 3.13
- NumPy (< 2), SciPy, Pandas
- Matplotlib, Plotly, cmcrameri
- psutil, joblib

---

## 🚀 Quick Start

### Define a velocity model

```python
import laytracer
import numpy as np
import pandas as pd

vel_df = pd.DataFrame({
    "Depth": [0.0, 1000.0, 2000.0, 3500.0],
    "Vp":    [3000.0, 4500.0, 5500.0, 6500.0],
    "Vs":    [1500.0, 2250.0, 2750.0, 3250.0],
    "Rho":   [2200.0, 2500.0, 2700.0, 2900.0],
    "Qp":    [200.0,  400.0,  600.0,  800.0],
    "Qs":    [100.0,  200.0,  300.0,  400.0],
})
```

### Trace a single ray

```python
source = np.array([0.0, 0.0, 3000.0])
receiver = np.array([5000.0, 1000.0, 0.0])
result = laytracer.trace_rays(
    source,
    receiver,
    vel_df,
    requested={"travel_times", "rays", "ray_parameters"},
)

print(f"Travel time:   {result.travel_times[0]:.4f} s")
print(f"Ray parameter: {result.ray_parameters[0]:.6e} s/m")
```

### Prescribe a converted reflection and request derivatives

```python
itinerary = laytracer.RayItinerary(
    source_phase="P",
    interactions=[
        laytracer.Interaction(3500.0, "reflect", "SV"),
    ],
)

converted = laytracer.trace_rays(
    source,
    receiver,
    vel_df,
    itinerary=itinerary,
    requested={
        "travel_times",
        "rays",
        "ray_parameters",
        "diagnostics",
        "sensitivities",
    },
)

certificate = converted.diagnostics[0]
kernel = converted.sensitivities[0]
print(certificate.method, certificate.absolute_offset_residual)
print(kernel.vp_layer_indices, kernel.dtravel_time_dvp)
print(kernel.vs_layer_indices, kernel.dtravel_time_dvs)
```

The itinerary is fixed by the caller. LayTracer does not search for phase
branches or guarantee a global first arrival. The legacy ``reflection=`` and
``refraction=`` tuple arguments remain available; in that API ``refraction``
means a prescribed transmission or mode conversion, not a head wave.

### Trace multiple rays in 3-D (with relative amplitude attributes)

```python
src = np.array([0.0, 0.0, 3000.0])
rcvs = np.column_stack([
    np.arange(500, 15001, 500),
    np.zeros(29),
    np.zeros(29),
])

result = laytracer.trace_rays(
    sources=src,
    receivers=rcvs,
    velocity_df=vel_df,
    source_phase="P",  # also accepts "SV", "SH", "S", or a list such as ["P", "SH", "SV"]
    requested={"travel_times", "rays", "ray_parameters", "tstar", "spreading", "trans_product"},
    transcoef_method="standard",  # standard Zoeppritz coefficients without normalization
)

# Access results
print(result.travel_times)   # travel times (s)
print(result.tstar)          # attenuation operator t*
print(result.spreading)      # geometrical spreading
print(result.trans_product)  # nonnegative coefficient-magnitude product
```

### Visualise results

```python
# 2-D ray paths over velocity cross-section
laytracer.plot.rays_2d(vel_df, rays=[r.ray_path for r in ...])

# 1-D velocity profile
laytracer.plot.velocity_profile(vel_df, param="Vp")

# Interactive 3-D viewer
fig = laytracer.plot.rays_3d(vel_df, rays=result.rays, sources=src, receivers=rcvs)
fig.show()
```

---

## 📐 API Overview

### Model

| Symbol | Description |
| --- | --- |
| `LayerStack` | Data class holding layer thicknesses, velocities (Vp, Vs), densities, and Q-factors |
| `ModelArrays` | Pre-extracted NumPy arrays from a velocity DataFrame for efficient repeated tracing |
| `build_layer_stack(vel_model, z_src, z_rcv)` | Extract the traversed layer stack between source and receiver depths (accepts DataFrame or `ModelArrays`) |

### Solver

| Symbol | Description |
| --- | --- |
| `solve(h, v, segments, interactions, ...)` | Low-level two-point solve for a compiled path |
| `RayResult` | Result container: travel time, ray path, ray parameter, optional attributes, and diagnostics |
| `SolveDiagnostics` | Solve route, iteration counts, transformed and physical parameters, endpoint residual, conditioning, and criticality margin |
| `offset(q, h, lmd)` | Total horizontal offset *X(q)* |
| `offset_dq(q, h, lmd)` | First derivative d*X*/d*q* |
| `offset_dq2(q, h, lmd)` | Second derivative d²*X*/d*q*² |
| `q_from_p(p, vmax)` / `p_from_q(q, vmax)` | Convert between slowness *p* and dimensionless *q* |
| `initial_q(X_target, h, lmd)` | Asymptotic initial estimate for Newton iteration |
| `newton_step(q_i, X_target, h, lmd)` | One quadratic Newton step |

### Amplitude

| Symbol | Description |
| --- | --- |
| `psv_rt_coefficients(p, vp1, vs1, rho1, vp2, vs2, rho2)` | All 8 P-SV reflection/transmission coefficients (Zoeppritz) |
| `sh_rt_coefficients(p, vs1, rho1, vs2, rho2)` | Decoupled SH-SH reflection/transmission coefficients |
| `normalize_rt_coefficient(coeff, p, v_in, rho_in, v_out, rho_out)` | Energy-flux normalization of R/T coefficients (Červený, 2001) |
| `find_brewster_angles(rt_coefficients, angles, ...)` | Detect Brewster-like zeros in R/T curves |

### Multi-ray

| Symbol | Description |
| --- | --- |
| `trace_rays(sources, receivers, velocity_df, ...)` | Trace all source–receiver pairs with optional parallelism; multi-phase input returns a phase-keyed result dictionary |
| `Interaction` / `RayItinerary` | Ordered, fixed-topology reflection and transmission/mode-conversion description |
| `TraceResult` | Container: source phase, travel times, ray paths, ray parameters, optional attributes, diagnostics, and sensitivities |
| `RaySensitivity` | Sparse fixed-topology derivatives with respect to layer velocities, interface depths, and endpoint coordinates |

### Visualisation (`laytracer.plot`)

| Function | Description |
| --- | --- |
| `velocity_profile(vel_df, ...)` | 1-D velocity–depth step profile (matplotlib) |
| `rays_2d(vel_df, rays, ...)` | 2-D ray paths over layered velocity cross-section (matplotlib) |
| `rays_3d(vel_df, rays, ...)` | Interactive 3-D ray visualisation (Plotly) |

---

## 📖 Documentation

Full documentation is built with [Sphinx](https://www.sphinx-doc.org) and includes:

- **Getting Started** — installation and environment setup
- **Methodology** — complete mathematical derivations (dimensionless parameter, Newton iteration, travel time, *t\**, geometrical spreading, Zoeppritz coefficients, critical & Brewster angles, 3-D extension)
- **Examples Gallery** — runnable scripts rendered with [Sphinx-Gallery](https://sphinx-gallery.github.io)
- **API Reference** — auto-generated from docstrings with [numpydoc](https://numpydoc.readthedocs.io)

### Build the docs

```bash
conda activate laytracer
# Windows
build-docs.bat
# Linux / macOS
chmod +x build-docs.sh
./build-docs.sh
```

Build docs with PDF output:

```bash
conda activate laytracer
# Windows
build-docs.bat -pdf
# Linux / macOS
chmod +x build-docs.sh
./build-docs.sh -pdf
```

You can do also using `make` commands:

```bash
conda activate laytracer
cd docs
# Build HTML
make html
# Build PDF
make latexpdf
# Run a local server to view HTML
cd build/html
python -m http.server
```

### Automatic docs deployment to GitHub Pages

This repository includes a GitHub Actions workflow at `.github/workflows/docs.yml` that runs on every push to `main` and:

- builds Sphinx HTML docs,
- builds the PDF (`laytracer.pdf`),
- copies the PDF into the published site (`_static/laytracer.pdf`),
- deploys HTML docs to GitHub Pages,
- uploads the PDF as a workflow artifact.

One-time GitHub setup:

1. Open **Settings → Pages** in your GitHub repository.
2. Set **Source** to **GitHub Actions**.
3. Push to `main`.

Published docs URL:

[https://danikiev.github.io/LayTracer](https://danikiev.github.io/LayTracer/)

---

## 🔬 Theory

LayTracer implements the method of [Fang & Chen (2019)](https://doi.org/10.1111/1365-2478.12799) for two-point ray tracing in horizontally layered media:

1. **Dimensionless ray parameter** *q* = *p* · *v*_max / √(1 − *p*² · *v*²_max) maps the full range of take-off angles to [0, ∞) without singularities.

2. **Offset equation** *X*(*q*) is a smooth, monotonically increasing function — ideal for Newton iteration.

3. **Safeguarded transformed-parameter solver** attempts rapid Newton iteration, falls back to a checked bracketed solve when needed, and accepts a solution only after independently recomputing the endpoint offset.

4. **Amplitude attributes** are computed inline:
   - Travel time from vertical slowness summation
   - Attenuation *t\** from per-layer *Q*-factors
   - Geometrical spreading from analytic ∂*X*/∂*p*
   - Full P-SV R/T coefficients and decoupled SH-SH coefficients (Lay & Wallace (1995) formulation)

### Key references

- Fang, X. & Chen, X. (2019). *A fast and robust two-point ray tracing method in layered media with constant or linearly varying layer velocity.* Geophysical Prospecting, 67(7), 1811–1824. [doi:10.1111/1365-2478.12799](https://doi.org/10.1111/1365-2478.12799)
- Aki, K. & Richards, P.G. (2002). *Quantitative Seismology.* 2nd ed., University Science Books.
- Lay, T. & Wallace, T.C. (1995). *Modern Global Seismology.* Academic Press.
- Červený, V. (2001). *Seismic Ray Theory.* Cambridge University Press. [doi:10.1017/CBO9780511529399](https://doi.org/10.1017/CBO9780511529399)

---

## 🧪 Testing

LayTracer includes a comprehensive test suite covering the solver, amplitude calculations, API, and physical symmetries.

```bash
conda activate laytracer
pytest
```

Test modules:

- `test_solver.py` — Newton convergence, Snell's law, travel time accuracy
- `test_amplitude.py` — Zoeppritz coefficients, energy-flux normalization, Brewster detection
- `test_api.py` — multi-ray tracing interface
- `test_sensitivity.py` — analytic traveltime and ray-parameter derivatives
- `test_generalized.py` — generalized layered-media validation cases
- `test_homogeneous_equivalence.py` — homogeneous-medium equivalence checks
- `test_plot.py` — plotting helpers and visualisation options
- `test_symmetry.py` — reciprocity and physical consistency checks

---

## 📂 Project Structure

```text
LayTracer/
├── laytracer/               # Main package
│   ├── __init__.py          # Public API exports
│   ├── model.py             # LayerStack, ModelArrays, build_layer_stack
│   ├── solver.py            # Core ray tracing solver (q-parameter + Newton)
│   ├── sensitivity.py       # Sparse fixed-topology analytic sensitivities
│   ├── amplitude.py         # Transmission coefficients, Zoeppritz, Brewster
│   ├── api.py               # High-level multi-ray interface (trace_rays)
│   └── plot.py              # Visualisation (2-D, 3-D, velocity profiles)
├── examples/                # Sphinx-Gallery example scripts
│   ├── 01_basic_raytracing.py
│   ├── 02_paper_examples.py
│   ├── 03_reflection_transmission.py
│   ├── 04_amplitude_analysis.py
│   ├── 05_homogeneous_equivalence.py
│   └── README.txt
├── branding/                # Branding assets and logo source
│   └── logo/                # Logo generator, vendored Poppins font, and usage docs
├── pytests/                 # Test suite
│   ├── test_solver.py
│   ├── test_amplitude.py
│   ├── test_api.py
│   ├── test_sensitivity.py
│   ├── test_generalized.py
│   ├── test_homogeneous_equivalence.py
│   ├── test_plot.py
│   └── test_symmetry.py
├── docs/                    # Sphinx documentation
│   └── source/
|       ├── api/                # Generated API reference pages
│       |   └── index.rst       # API reference landing page
|       ├── changelog.rst       # Changelog page with a placeholder for changelog content generated from CHANGELOG.md
│       ├── citing.rst          # Citation information and BibTeX entries
│       ├── conf.py             # Sphinx configuration
│       ├── credits.rst         # Author and contributor acknowledgements and library credits
│       ├── getting_started.rst # Installation and quick start
│       ├── index.rst           # Main index page with links to all sections
│       ├── methodology.rst     # Full mathematical derivations
│       └── references.bib      # Bibliography for the docs
├── pyproject.toml           # Build configuration (setuptools + setuptools-scm)
├── environment.yml          # Conda environment specification
├── pytest.ini               # Pytest configuration
├── CHANGELOG.md             # Changelog
├── LICENSE                  # MIT License file
├── RELEASE.md               # Release guidelines and GitHub Actions workflow documentation
└── README.md                # This README file 
```

---

## 📄 License

LayTracer is released under the [MIT License](LICENSE).

---

## 👤 Author

[**Denis Anikiev**](@danikiev) — [danikiev@gmail.com](mailto:danikiev@gmail.com)

---

## 📝 Citation

If you use a LayTracer in your research, please cite:

```bibtex
@software{Anikiev2026LayTracer,
  author       = {Anikiev, Denis},
  title        = {{LayTracer}: {F}ast two-point seismic ray tracing in layered media},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18850919},  
  url          = {https://github.com/danikiev/LayTracer}
}
```

The DOI in the above citation points to the latest version of LayTracer.
If you use a specific version, please cite that version's DOI (available on the Zenodo release page), e.g.:

```bibtex
@software{Anikiev2026LayTracerVersion,
  author       = {Anikiev, Denis},
  title        = {{LayTracer}: {F}ast two-point seismic ray tracing in layered media},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {0.4.0},    
  license      = {MIT},
  doi          = {10.5281/zenodo.19983599},
  url          = {https://github.com/danikiev/LayTracer}
}
```
