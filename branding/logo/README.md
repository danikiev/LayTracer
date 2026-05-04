# LayTracer Logo Assets

This folder contains the source code and vendored font needed to regenerate the LayTracer SVG logo assets.

Regenerate the committed SVG files from the LayTracer repository root:

```powershell
conda run -n laytracer python branding/logo/generate_logos.py
```

Check whether the committed assets are up to date:

```powershell
conda run -n laytracer python branding/logo/generate_logos.py --check
```

The Poppins font is licensed under the SIL Open Font License 1.1; see `fonts/OFL.txt`.
