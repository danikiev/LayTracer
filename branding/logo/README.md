# LayTracer Logo Assets

This folder contains the source code and vendored Poppins font needed to regenerate the LayTracer logo assets. The generated SVG and PDF files are committed under `docs/source/_static/` so they can be used directly by the README and Sphinx documentation.

Run from the LayTracer repository root:

```powershell
conda run -n laytracer python branding/logo/generate_logos.py
```

Check whether the committed logo files are up to date:

```powershell
conda run -n laytracer python branding/logo/generate_logos.py --check
```

## Design Idea

The LayTracer logo is a compact visual summary of two-point ray tracing in layered media. The Poppins wordmark keeps the brand clean and technical: `Lay` is set in dark navy to suggest the stable layered model, while `Tracer` is set in accent orange to emphasize the active tracing operation.

The thick light gray-blue underline represents a layer boundary or interface. The lowered `y` in `Lay` turns the word itself into part of the ray diagram: its two upper branches act as the incident and reflected ray paths, meeting at the interface. The dashed light gray-blue overlays follow those branches to make the ray interpretation explicit without replacing the original glyph shape.

The orange point marks the source-side ray branch, and the dark navy point marks the receiver/reflected branch. In the full logo, the tagline is split around the descending branch of the `y`: `FAST TWO-POINT` is orange, while `SEISMIC RAY TRACING IN LAYERED MEDIA` remains navy, matching the wordmark color logic.

## Colors

- Dark navy `#0B1F3B`: `Lay`, receiver/reflected point, and navy tagline text.
- Accent orange `#FF6A1A`: `Tracer`, source point, and `FAST TWO-POINT` in the full logo tagline.
- Light gray-blue `#9AA7B2`: layer-boundary interface line and dashed ray overlays.
- White `#FFFFFF`: background fill for the circular icon variant.

## Outputs

The generator writes:

- `docs/source/_static/laytracer-logo-full.svg`
- `docs/source/_static/laytracer-logo-full.pdf`
- `docs/source/_static/laytracer-logo-medium.svg`
- `docs/source/_static/laytracer-icon.svg`
- `docs/source/_static/laytracer-icon-circle.svg`

## Usage

- `laytracer-logo-full.svg` is shown near the top of the repository `README.md`.
- `laytracer-logo-full.svg` is shown on the HTML documentation index page after the `Overview` heading.
- `laytracer-logo-full.pdf` is the PDF-compatible full logo used on the LaTeX title page.
- `laytracer-logo-medium.svg` is used as the Sphinx/pydata theme header logo in `docs/source/conf.py`.
- `laytracer-icon-circle.svg` is used as the Sphinx favicon in `docs/source/conf.py`.
- `laytracer-icon.svg` is the plain icon variant for contexts that do not need a circular background.

Both icon variants use the same generated `y` motif geometry as the full and medium logos. The interface line length is shared between the plain and circular icons; in the circular variant it is computed as a chord inside the circle.

The Poppins font is licensed under the SIL Open Font License 1.1; see `fonts/OFL.txt`.
