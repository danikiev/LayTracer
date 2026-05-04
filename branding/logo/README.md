# LayTracer Logo Assets

This folder contains the source code and vendored Poppins font needed to regenerate the LayTracer SVG logo assets. The generated SVG files are committed under `docs/source/_static/` so they can be used directly by the README and Sphinx documentation.

Run from the LayTracer repository root:

```powershell
conda run -n laytracer python branding/logo/generate_logos.py
```

Check whether the committed SVG files are up to date:

```powershell
conda run -n laytracer python branding/logo/generate_logos.py --check
```

## Outputs

The generator writes:

- `docs/source/_static/laytracer-logo-full.svg`
- `docs/source/_static/laytracer-logo-medium.svg`
- `docs/source/_static/laytracer-icon.svg`

## Usage

- `laytracer-logo-full.svg` is shown near the top of the repository `README.md`.
- `laytracer-logo-medium.svg` is used as the Sphinx/pydata theme header logo in `docs/source/conf.py`.
- `laytracer-icon.svg` is used as the Sphinx favicon in `docs/source/conf.py`.

The Poppins font is licensed under the SIL Open Font License 1.1; see `fonts/OFL.txt`.
