# Release Guide

This project uses `setuptools_scm`, so the package version is derived from Git tags.
Use semantic tags in the form `vX.Y.Z` (example: `v0.1.1`). A pre-release appends a PEP 440 segment `aN`, `bN` or `rcN` (example: `v1.0.0rc1`); see [Pre-releases](#pre-releases).

## 1. One-time setup

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

## 2. Standard release flow (TestPyPI first)

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

## 3. Quick automation commands

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

## 4. Common errors

- `403 Forbidden`: wrong token or wrong repository (PyPI token vs TestPyPI token).
- `File already exists`: this version is already uploaded; bump tag/version and rebuild.
- Unexpected dev version: release tag missing or not fetched in current Git checkout.
- Live docs show a dev version or an older release after a release: see [Deploying the docs manually](#deploying-the-docs-manually).

## 5. Tag-push docs automation (`push-tag.yml`)

This repository includes `.github/workflows/push-tag.yml`.

Behavior:

- Push `v*` tag -> validate tag -> start a `Docs` run on `main` with `ref` set to the tag -> that run builds the documentation of the tag, deploys it, and checks that the live site shows its version.
- Package publishing is intentionally not done here; it is handled by `release-on-published.yml`.
- Tag validation is shared through `.github/workflows/validate-tag.yml`.

Only `Docs` runs on `main` deploy. GitHub Pages ignores a deployment made from a tag when `main` has already deployed the same commit, and still reports it as successful ([actions/deploy-pages#383](https://github.com/actions/deploy-pages/issues/383)). A release tag sits on a commit that `main` deployed when it was merged, so while tags deployed their own docs, the v0.5.0 site kept showing `0.4.1.dev38+gf9703a9be` although the docs of the tag had built correctly.

Required GitHub setup:

1. Repository environment `github-pages`: deployments come only from `main`, which the environment allows by default. The `v*` tag rule in **Settings → Environments → github-pages → Deployment branches and tags**, which tag deployments needed, is no longer used.

2. Trigger a docs deployment by tag:

```powershell
git checkout main
git pull
git tag v0.1.2
git push origin v0.1.2
```

Notes:

- The workflow fetches tags so `setuptools_scm` can resolve the release version correctly.
- Release tags must point to a commit contained in `main`; tags created from `dev` or any other branch are rejected by the workflow.
- Only tags in the form `vMAJOR.MINOR.PATCH` (example: `v1.2.3`) are accepted, optionally followed by a pre-release segment `aN`, `bN` or `rcN` (example: `v1.0.0rc1`).

### Deploying the docs manually

Use this when the live site shows the wrong version, for example when the `Check the deployed version` step of a `Docs` run failed, or to republish the docs of a release:

1. On GitHub, open **Actions → Docs → Run workflow**.
2. Keep **Use workflow from** at `main`. Runs on any other branch or on a tag build the docs but do not deploy them.
3. Set **ref** to the release tag, for example `refs/tags/v0.5.0`, or leave it empty to publish the current `main`.
4. Click **Run workflow**. The run builds the docs from **ref**, checks that the docs of a tag carry its version, deploys them and checks the live site.

The same from a terminal, with the GitHub CLI:

```powershell
gh workflow run docs.yml --ref main -f ref=refs/tags/v0.5.0
```

To see which version the live site serves (the random query string bypasses the CDN cache):

```powershell
curl.exe -s "https://danikiev.github.io/LayTracer/_static/documentation_options.js?$(Get-Random)" | Select-String VERSION
```

## 6. Release-event automation (`release-on-published.yml`)

This repository also includes `.github/workflows/release-on-published.yml`.

Behavior:

- Trigger: GitHub Release event `published` (plus optional `workflow_dispatch` with a tag input).
- Flow: build package -> publish to TestPyPI -> wait for approval -> publish same artifacts to PyPI.
- The published release tag must point to a commit contained in `main`; releases created from tags on other branches are rejected during validation.
- Zenodo metadata validation must pass before package build/publish starts.
- Tag validation is shared through `.github/workflows/validate-tag.yml`.

Required GitHub setup:

1. Add repository secrets:
    - `TEST_PYPI_API_TOKEN` = token from TestPyPI
    - `PYPI_API_TOKEN` = token from PyPI

2. Create repository environments:
    - `testpypi` (optional protection rules)
    - `pypi` (recommended: require reviewers for manual approval gate)

Release order:

- Push the `vX.Y.Z` tag first to deploy versioned docs.
- Publish the GitHub Release for that tag when you are ready to publish to TestPyPI/PyPI.
- Both workflows accept only `vMAJOR.MINOR.PATCH` tags, optionally with a pre-release segment (`aN`, `bN`, `rcN`), before doing release work.

### Pre-releases

A pre-release, such as a release candidate, follows the same flow with a pre-release tag:

- Write the tag in canonical PEP 440 form: `v1.0.0a1`, `v1.0.0b1` or `v1.0.0rc1`. The build checks that the distribution files carry the version of the tag, and `setuptools_scm` normalizes other spellings: `v1.0.0-rc.1` would build `1.0.0rc1`. Tag validation therefore rejects such spellings.
- Select **Set as a pre-release** when you create the GitHub Release. Its `published` event still starts `release-on-published.yml`, which uploads the pre-release to TestPyPI and PyPI.
- `pip install laytracer` skips pre-releases. Install one with an explicit version, `pip install laytracer==1.0.0rc1`, or with `pip install --pre laytracer`.
- The tag push deploys the documentation of the pre-release, so the live site shows its version until the next release.

## 7. Zenodo release plan (recommended order)

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
    - Push the release tag (`vMAJOR.MINOR.PATCH`, or `vMAJOR.MINOR.PATCHrcN` for a release candidate).
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
