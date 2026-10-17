r"""Tests that the public API documents its parameters, returns and fields."""

import dataclasses
import inspect
import re

import pytest

import laytracer

_SECTION = re.compile(r"^(?P<title>[A-Z][A-Za-z ]+)\n-{3,}\n", re.M)
_ENTRY = re.compile(r"^([A-Za-z_*][\w*]*(?:\s*,\s*[A-Za-z_*][\w*]*)*)\s*:")


def _documented_names(doc, titles):
    """Return the names listed in the numpydoc sections *titles* of *doc*."""
    names = set()
    sections = list(_SECTION.finditer(doc))
    for index, section in enumerate(sections):
        if section.group("title") not in titles:
            continue
        end = sections[index + 1].start() if index + 1 < len(sections) else len(doc)
        for line in doc[section.end():end].splitlines():
            entry = _ENTRY.match(line)
            if entry:
                names.update(name.strip().lstrip("*") for name in entry.group(1).split(","))
    return names


def _public(predicate):
    return [name for name in laytracer.__all__ if predicate(getattr(laytracer, name))]


@pytest.mark.parametrize("name", _public(inspect.isfunction))
def test_public_function_documents_parameters_and_returns(name):
    """Every argument appears under Parameters, and a Returns section exists."""
    func = getattr(laytracer, name)
    doc = inspect.getdoc(func) or ""
    missing = set(inspect.signature(func).parameters) - _documented_names(doc, {"Parameters"})

    assert not missing, f"{name} does not document {sorted(missing)}"
    assert re.search(r"^(Returns|Yields)\n-{3,}$", doc, re.M), f"{name} has no Returns section"


@pytest.mark.parametrize("name", _public(dataclasses.is_dataclass))
def test_public_dataclass_documents_fields(name):
    """Every field appears under Parameters or Attributes."""
    cls = getattr(laytracer, name)
    doc = inspect.getdoc(cls) or ""
    fields = {field.name for field in dataclasses.fields(cls)}
    missing = fields - _documented_names(doc, {"Parameters", "Attributes"})

    assert not missing, f"{name} does not document {sorted(missing)}"
