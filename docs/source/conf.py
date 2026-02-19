# -*- coding: utf-8 -*-
import sys
import os
import datetime
from sphinx_gallery.sorting import ExampleTitleSortKey
import plotly.io as pio

pio.renderers.default = 'sphinx_gallery'

# Sphinx needs to be able to import the package
sys.path.insert(0, os.path.abspath("../../laytracer"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_design",    
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "plotly": ('https://plotly.com/python-api-reference/', None),    
}

# Generate autodoc stubs with summaries from code
autosummary_generate = True

# Include Python objects as they appear in source files
autodoc_member_order = "bysource"

# Default flags used by autodoc directives
autodoc_default_flags = ["members"]

# Avoid showing typing annotations in doc
autodoc_typehints = "none"

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

# Gallery configuration
sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        "../../examples",        
    ],
    # path where to save gallery generated examples
    "gallery_dirs": ["examples"],
    #"tutorials"],
    "filename_pattern": r"\.py", #include all examples
    #"filename_pattern": r"02_amplitude_analysis.py", #include specific example only
    # Ignore pattern to exclude specific files (useful to exclude demanding examples)
    #"ignore_pattern": r"\03_*\.py$", #exclude by mask
    #"ignore_pattern": r"\.py$", # exclude all examples
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Sort gallery examples by file name instead of number of lines (default)
    "within_subsection_order": ExampleTitleSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": "api/generated/backreferences",
    # Modules for which function level galleries are created.
    "doc_module": "laytracer",
    # Insert links to documentation of objects in the examples
    "reference_url": {"laytracer": None},
    # Add scraper for plotly
    # 'image_scrapers': ('matplotlib',plotly_sg_scraper),    
}

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ["png"]

# Sphinx project configuration
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "**.ipynb", "**.md5"]
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8-sig"
master_doc = "index"

# Version
version = 'X' # __version__
if len(version.split("+")) > 1 or version == "unknown":
    version = "dev"

# General
author = "Denis Anikiev"
year = datetime.date.today().year
project = "LayTracer"
copyright = f"{year}, {author}"


# These enable substitutions using |variable| in the rst files
rst_epilog = """
.. |year| replace:: {year}
""".format(
    year=year
)
html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"
html_title = "LayTracer"
html_short_title = "LayTracer"
#html_logo = "_static/laytracer.png"
#html_favicon = "_static/favicon.ico"
html_extra_path = []
pygments_style = "default"
add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# Check if on GitHub CI
on_github_ci = os.environ.get("GITHUB_ACTIONS") == "true"

if on_github_ci:
    pdf_url = "/laytracer/_static/laytracer.pdf"
else:
    pdf_url = "/_static/laytracer.pdf"

# Theme config
html_theme = "pydata_sphinx_theme"
html_theme_options = {
#     "logo_only": True,
#     "display_version": True,
#     "logo": {
#         "image_light": "logo.png",
#         "image_dark": "logo.png",
#     }
    "icon_links": [        
        {         
            "name": "GitHub",         
            "url": "https://github.com/danikiev/LayTracer",
            "icon": "fab fa-github",            
            "type": "fontawesome",
        },
        {            
            "name": "Download as PDF",
            "url": pdf_url,            
            "icon": "fas fa-file-pdf",
            "type": "fontawesome",
        }        
    ],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "show_nav_level": 3,
    "navigation_depth": 3,
    "collapse_navigation": True,
}

# Custom CSS
html_css_files = [
    'css/custom.css',    
]

# Custom sidebar settings for specific pages
html_sidebars = {    
    'overview': [],        # Hide sidebar on the overview page
    'getting_started': [], # Hide sidebar on the getting_started page
    'methodology': [],     # Hide sidebar on the methodology page
    'contributing': [],    # Hide sidebar on the contributing page
    'changelog': [],       # Hide sidebar on the changelog page
    'citing': [],          # Hide sidebar on the citing page
    'credits': []          # Hide sidebar on the credits page
}

html_context = {
    "menu_links_name": "Repository",    
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    "doc_path": "docs/source",
    "galleries": sphinx_gallery_conf["gallery_dirs"],
    "gallery_dir": dict(
        zip(sphinx_gallery_conf["gallery_dirs"], sphinx_gallery_conf["examples_dirs"])
    ),
    "github_project": "LayTracer",
    "github_repo": "LayTracer",
    "github_version": "main",
}

# Load the custom CSS files (needs sphinx >= 1.6 for this to work)
def setup(app):
    app.add_css_file("style.css")

# bibtex
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# Latex Setup for conversion to PDF
latex_elements = {
    'preamble': r'''
\usepackage{csquotes}
\usepackage[style=authoryear,backend=biber]{biblatex}
\addbibresource{references.bib}
\usepackage[titles]{tocloft}
\addto\captionsenglish{\renewcommand{\bibname}{References}}
\AtBeginBibliography{\small}
    ''',
}

# Use more advanced LaTeX to deal with unicode characters
#latex_engine = 'xelatex'
latex_engine = 'lualatex'

# Add fontawesome icons in PDF 
#sd_fontawesome_latex = True # somehow not working (already loaded)

