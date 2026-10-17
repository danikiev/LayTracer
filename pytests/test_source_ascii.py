r"""Tests that the package source is plain ASCII."""

from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "laytracer"


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=lambda path: path.name)
def test_package_source_is_ascii(path):
    """No package source file contains a non-ASCII character."""
    lines = path.read_text(encoding="utf-8").splitlines()
    offending = [(number, line.strip()) for number, line in enumerate(lines, 1) if not line.isascii()]

    assert not offending, f"non-ASCII text in {path.name}: {offending[:5]}"
