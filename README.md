# LayTracer

[![Pytest](https://github.com/danikiev/LayTracer/actions/workflows/pytest.yml/badge.svg)](https://github.com/danikiev/LayTracer/actions/workflows/pytest.yml)
[![Docs](https://github.com/danikiev/LayTracer/actions/workflows/docs.yml/badge.svg)](https://github.com/danikiev/LayTracer/actions/workflows/docs.yml)

**Fast two-point seismic ray tracing in 1-D layered media.**

LayTracer is a Python package for computing ray paths, travel times, and amplitude attributes in horizontally layered velocity models with constant layer velocities. It implements the dimensionless ray parameter method of [Fang & Chen (2019)](https://doi.org/10.1111/1365-2478.12799), achieving rapid convergence.

---

## ✨ Features

| Category | Capability |
| :------- | :--------- |
| **Ray tracing** | Second-order (quadratic) Newton solver using the dimensionless *q*-parameter for robust, singularity-free convergence |
| **Travel time** | Layer-by-layer travel time summation from the solved ray parameter |
| **Attenuation** | Intrinsic absorption operator *t\** from quality factors *Q* |
| **Spreading** | Relative geometrical spreading from the analytical ray-tube Jacobian ∂X/∂p |
| **Transmission** | Normal-incidence (impedance) and full angle-dependent Zoeppritz P-SV coefficients (all 8 R/T modes) |
| **Brewster angles** | Automatic detection of Brewster-like zeros in R/T coefficient curves |
| **Parallel execution** | Multi-ray tracing with `joblib` / `loky` backend for large surveys |
| **Visualisation** | 2-D ray path plots (matplotlib) and interactive 3-D viewer (Plotly) |
| **Documentation** | Comprehensive Sphinx docs with extensive theory, gallery examples, and API reference |

---

## 📦 Installation

### Prerequisites

- [Conda](https://conda.io) package manager (we recommend [miniforge](https://github.com/conda-forge/miniforge))

### Install with conda

```bash
# 1. Create environment with all dependencies
conda env create -f environment.yml

# 2. Activate it
conda activate laytracer

# 3. Install LayTracer in editable mode
pip install -e .
```

### Quick install (Windows)

```batch
install.bat
```

### Quick install (Linux / macOS)

```bash
chmod +x install.sh
./install.sh
```

### Dependencies

- Python ≥ 3.8, < 3.13
- NumPy, SciPy, Pandas
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

### Trace a single 2-D ray

```python
stack = laytracer.build_layer_stack(vel_df, z_src=3000.0, z_rcv=0.0)

result = laytracer.solve(
    stack,
    epicentral_dist=5000.0,
    z_src=3000.0,
    z_rcv=0.0,
    vel_type="Vp",
)

print(f"Travel time:   {result.travel_time:.4f} s")
print(f"Ray parameter: {result.ray_parameter:.6e} s/m")
```

### Trace multiple rays in 3-D (with amplitude)

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
    vel_type="Vp",
    compute_amplitude=True,
    transcoef_method="angle",  # full Zoeppritz
)

# Access results
print(result.travel_times)   # travel times (s)
print(result.tstar)          # attenuation operator t*
print(result.spreading)      # geometrical spreading
print(result.trans_product)  # product of transmission coefficients
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
|---|---|
| `LayerStack` | Data class holding layer thicknesses, velocities (Vp, Vs), densities, and Q-factors |
| `build_layer_stack(vel_df, z_src, z_rcv)` | Extract the traversed layer stack between source and receiver depths |

### Solver

| Symbol | Description |
|---|---|
| `solve(stack, epicentral_dist, ...)` | Solve the two-point ray tracing problem for one source–receiver pair |
| `RayResult` | Result container: travel time, ray path, ray parameter, *t\**, spreading, transmission product |
| `offset(q, h, lmd)` | Total horizontal offset *X(q)* |
| `offset_dq(q, h, lmd)` | First derivative d*X*/d*q* |
| `offset_dq2(q, h, lmd)` | Second derivative d²*X*/d*q*² |
| `q_from_p(p, vmax)` / `p_from_q(q, vmax)` | Convert between slowness *p* and dimensionless *q* |
| `initial_q(X_target, h, lmd)` | Asymptotic initial estimate for Newton iteration |
| `newton_step(q_i, X_target, h, lmd)` | One quadratic Newton step |

### Amplitude

| Symbol | Description |
|---|---|
| `transmission_normal(v1, rho1, v2, rho2)` | Normal-incidence displacement transmission coefficient |
| `psv_rt_coefficients(p, vp1, vs1, rho1, vp2, vs2, rho2)` | All 8 P-SV reflection/transmission coefficients (Zoeppritz) |
| `find_brewster_angles(rt_coefficients, angles, ...)` | Detect Brewster-like zeros in R/T curves |

### Multi-ray

| Symbol | Description |
|---|---|
| `trace_rays(sources, receivers, velocity_df, ...)` | Trace all source–receiver pairs with optional parallelism |
| `TraceResult` | Container: travel times, ray paths, ray parameters, *t\**, spreading, transmission products |

### Visualisation (`laytracer.plot`)

| Function | Description |
|---|---|
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

Published docs URL pattern:

`https://<username>.github.io/<repository>/`

---

## 🔬 Theory

LayTracer implements the method of [Fang & Chen (2019)](https://doi.org/10.1111/1365-2478.12799) for two-point ray tracing in horizontally layered media:

1. **Dimensionless ray parameter** *q* = *p* · *v*_max / √(1 − *p*² · *v*²_max) maps the full range of take-off angles to [0, ∞) without singularities.

2. **Offset equation** *X*(*q*) is a smooth, monotonically increasing function — ideal for Newton iteration.

3. **Quadratic Newton solver** with asymptotic initial estimate converges in **2–3 iterations**.

4. **Amplitude attributes** are computed inline:
   - Travel time from vertical slowness summation
   - Attenuation *t\** from per-layer *Q*-factors
   - Geometrical spreading from analytic ∂*X*/∂*p*
   - Full Zoeppritz P-SV R/T coefficients (Lay & Wallace (1995) formulation)

### Key references

- Fang, X. & Chen, X. (2019). *A fast and robust two-point ray tracing method in layered media.* Geophysical Prospecting, 67(7), 1648–1661. [doi:10.1111/1365-2478.12799](https://doi.org/10.1111/1365-2478.12799)
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
- `test_amplitude.py` — Zoeppritz coefficients, normal-incidence limits, Brewster detection
- `test_api.py` — multi-ray tracing interface
- `test_generalized.py` — generalized layered-media validation cases
- `test_homogeneous_equivalence.py` — homogeneous-medium equivalence checks
- `test_symmetry.py` — reciprocity and physical consistency checks


---

## 📂 Project Structure

```
LayTracer/
├── laytracer/               # Main package
│   ├── __init__.py          # Public API exports
│   ├── model.py             # LayerStack, build_layer_stack
│   ├── solver.py            # Core ray tracing solver (q-parameter + Newton)
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
├── pytests/                 # Test suite
│   ├── test_solver.py
│   ├── test_amplitude.py
│   ├── test_api.py
│   ├── test_generalized.py
│   ├── test_homogeneous_equivalence.py
│   └── test_symmetry.py
├── docs/                    # Sphinx documentation
│   └── source/
│       ├── index.rst
│       ├── getting_started.rst
│       ├── methodology.rst   # Full mathematical derivations
│       └── references.bib
├── pyproject.toml           # Build configuration (setuptools + setuptools-scm)
├── environment.yml          # Conda environment specification
├── pytest.ini               # Pytest configuration
└── LICENSE                  # MIT License
```

---

## 📄 License

LayTracer is released under the [MIT License](LICENSE).

---

## 👤 Author

[**Denis Anikiev**](@danikiev) — [danikiev@gmail.com](mailto:danikiev@gmail.com)

---

## 📝 Citation

If you use LayTracer in your research, please cite:

```bibtex
@software{LayTracer,
  author  = {Anikiev, Denis},
  title   = {LayTracer: Fast two-point seismic ray tracing in 1-D layered media},
  url     = {https://github.com/danikiev/LayTracer},
  license = {MIT}
}
```
