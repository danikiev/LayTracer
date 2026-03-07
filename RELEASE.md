# Release Guide

This project uses `setuptools_scm`, so the package version is derived from Git tags.
Use semantic tags in the form `vX.Y.Z` (example: `v0.1.1`).

## 1) One-time setup

Install tools:

```powershell
python -m pip install --upgrade build twine
```

Create API tokens:

- TestPyPI token from [https://test.pypi.org](https://test.pypi.org)
- PyPI token from [https://pypi.org](https://pypi.org)

Optional: create `%USERPROFILE%\.pypirc` to avoid passing token on command line:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-TESTPYPI_TOKEN_HERE

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-PYPI_TOKEN_HERE
```

## 2) Standard release flow (TestPyPI first)

From repo root:

```powershell
git checkout main
git pull
```

Pick next version and tag it:

```powershell
git tag v0.1.1
git push origin v0.1.1
```

Build + validate:

```powershell
python -m build
python -m twine check dist/*
```

Upload to TestPyPI:

```powershell
python -m twine upload -r testpypi dist/*
```

Test install from TestPyPI:

```powershell
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple laytracer==0.1.1
python -c "import laytracer; print(laytracer.__version__)"
```

Upload to PyPI:

```powershell
python -m twine upload -r pypi dist/*
```

## 3) Quick automation commands

Use these two command blocks as a repeatable release routine.

### TestPyPI publish

```powershell
$version = "0.1.1"
git checkout main
git pull
git tag "v$version"
git push origin "v$version"
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
python -m twine upload -r testpypi dist/*
```

### PyPI publish (after TestPyPI verification)

```powershell
python -m twine upload -r pypi dist/*
```

## 4) Common errors

- `403 Forbidden`: wrong token or wrong repository (PyPI token vs TestPyPI token).
- `File already exists`: this version is already uploaded; bump tag/version and rebuild.
- Unexpected dev version: release tag missing or not fetched in current Git checkout.

## 5) GitHub Actions automation (`release.yml`)

This repository includes `.github/workflows/release.yml`.

Behavior:

- Push `v*` tag -> build package -> upload to TestPyPI -> wait for approval -> upload same artifacts to PyPI.
- `workflow_dispatch` is also available for manual runs.

Required GitHub setup:

1. Add repository secrets:
    - `TEST_PYPI_API_TOKEN` = token from TestPyPI
    - `PYPI_API_TOKEN` = token from PyPI

2. Create repository environments:
    - `testpypi` (optional protection rules)
    - `pypi` (recommended: require reviewers for manual approval gate)
    - `github-pages`: in **Settings → Environments → github-pages → Deployment branches and tags**, add a tag rule with pattern `v*` so that tag-triggered releases can deploy docs with the correct version.

3. Trigger a release by tag:

```powershell
git checkout main
git pull
git tag v0.1.2
git push origin v0.1.2
```

Notes:

- The workflow uses `fetch-depth: 0` so `setuptools_scm` can resolve the tag correctly.
- If PyPI publish fails with "File already exists", bump to a new tag and rerun.
- Publishing is guarded: only strict semver tags in the form `vMAJOR.MINOR.PATCH` are accepted (example: `v1.2.3`).
- Upload steps use `skip-existing: true`, so reruns with the same version do not fail if files are already present.

## 6) Release-event automation (`release-on-published.yml`)

This repository also includes `.github/workflows/release-on-published.yml`.

Behavior:

- Trigger: GitHub Release event `published` (plus optional `workflow_dispatch`).
- Flow: build package -> publish to TestPyPI -> wait for approval -> publish same artifacts to PyPI.

When to use which workflow:

- Use `release.yml` if you want publishing to start immediately on tag push (`git push origin vX.Y.Z`).
- Use `release-on-published.yml` if you want publishing only after explicitly creating/publishing a GitHub Release in the UI.
- Both workflows enforce strict semver tags (`vMAJOR.MINOR.PATCH`) before build/publish jobs run.

## 7) Zenodo release plan (recommended order)

Use this checklist to publish a citable Zenodo record for each tagged release.

1. One-time integration setup:
    - Sign in to [https://zenodo.org](https://zenodo.org) (or sandbox first: [https://sandbox.zenodo.org](https://sandbox.zenodo.org)).
    - In Zenodo GitHub settings, enable the `LayTracer` repository.
    - Verify the repository has a valid `.zenodo.json` metadata file.

### Zenodo DOI badge (before first release)

- Get the GitHub repository ID from: `https://api.github.com/repos/{user}/{repo}`.
- Add a DOI badge to your README before the first release.

Markdown (`README.md`):

```markdown
[![DOI](https://zenodo.org/badge/{github_id}.svg)](https://zenodo.org/badge/latestdoi/{github_id})
```

reStructuredText (`README.rst`):

```rst
|DOI|

.. |DOI| image:: https://zenodo.org/badge/{github_id}.svg
        :target: https://zenodo.org/badge/latestdoi/{github_id}
```

Example of a badge, this repository:

[![DOI](https://zenodo.org/badge/1160026484.svg)](https://zenodo.org/badge/latestdoi/1160026484)

Notes:

- The DOI badge appears only after the first published release, then points to the latest DOI.
- See GitHub documentation for Zenodo webhook/setup details.
- For this repository, DOI badge setup is already done; keep this section as reference for future migrations/new repos.

1. Pre-release metadata update:
    - Update `.zenodo.json` fields for the new release (`version`, `description`, `keywords`, `creators` as needed).
    - Ensure `LICENSE` and `README.md` are current and consistent with release notes.

2. Create the release in GitHub:
    - Push the semver tag (`vMAJOR.MINOR.PATCH`).
    - Create/publish a GitHub Release for that tag (include highlights from `CHANGELOG.md`).

3. Validate Zenodo deposition:
    - Confirm Zenodo auto-created a new versioned deposition from the GitHub release.
    - Check title, version, creators, license, and description in Zenodo UI.
    - Verify files are attached correctly (source archive and metadata).

4. Finalize and communicate DOI:
    - Publish/approve the Zenodo record (if not auto-published by your setup).
    - Copy version DOI and concept DOI into release notes and project docs (e.g., `README.md`).
    - Optionally add/update DOI badge in `README.md`.

5. Post-release verification:
    - Open the DOI link and ensure citation metadata resolves correctly.
    - Confirm the new version is discoverable under the LayTracer concept record.
