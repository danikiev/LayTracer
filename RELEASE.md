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

## 6) Release-event automation (`release-on-published.yml`)

This repository also includes `.github/workflows/release-on-published.yml`.

Behavior:

- Trigger: GitHub Release event `published` (plus optional `workflow_dispatch`).
- Flow: build package -> publish to TestPyPI -> wait for approval -> publish same artifacts to PyPI.

When to use which workflow:

- Use `release.yml` if you want publishing to start immediately on tag push (`git push origin vX.Y.Z`).
- Use `release-on-published.yml` if you want publishing only after explicitly creating/publishing a GitHub Release in the UI.
