# -*- coding: utf-8 -*-
import sys
import os
import datetime
from sphinx_gallery.sorting import ExampleTitleSortKey

# Sphinx needs to be able to import the package
sys.path.insert(0, os.path.abspath("../../laytracer"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
numpydoc_show_class_members = False

# Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["examples"],
    "filename_pattern": r"\.py",
    "download_all_examples": False,
    "within_subsection_order": ExampleTitleSortKey,
    "backreferences_dir": "api/generated/backreferences",
    "doc_module": "laytracer",
    "reference_url": {"laytracer": None},
}

plot_include_source = True
plot_formats = ["png"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
source_suffix = ".rst"
source_encoding = "utf-8-sig"
master_doc = "index"

# Version
version = "dev"

# General
author = "Denis Anikiev"
year = datetime.date.today().year
project = "LayTracer"
copyright = f"{year}, {author}"

html_static_path = ["_static"]
html_title = "LayTracer"
html_short_title = "LayTracer"
pygments_style = "default"
add_function_parentheses = False
html_show_sourcelink = False

# Theme
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 3,
    "navigation_depth": 3,
    "collapse_navigation": True,
}

html_sidebars = {
    "index": [],
    "theory": [],
}

# bibtex
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# LaTeX
latex_engine = "lualatex"
