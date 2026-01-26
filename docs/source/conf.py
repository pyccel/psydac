# pylint: disable=redefined-builtin

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# add these directories to sys.path here.

# Make AutosummaryRenderer have a filter (smart_fullname) that reduces psydac.module.submodule to module.submodule in the navigation part of the documentation
from sphinx.ext.autosummary.generate import AutosummaryRenderer

def smart_fullname(fullname):
    parts = fullname.split(".")
    return ".".join(parts[1:])

def fixed_init(self, app):
    AutosummaryRenderer.__old_init__(self, app)
    self.env.filters["smart_fullname"] = smart_fullname

AutosummaryRenderer.__old_init__ = AutosummaryRenderer.__init__
AutosummaryRenderer.__init__ = fixed_init

import pathlib
import sys
import tomli
import psydac

autodoc_mock_imports = [
    'gelato',
    'h5py',
    'matplotlib'
    'mpi4py',
    'numpy',
#    'psydac',
    'pyccel',
    'pyevtk',
    'scipy',
    'sympde',
    'sympy',
    'yaml',
]
#sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

with open('../../pyproject.toml', mode='rb') as pyproject:
    pkg_meta = tomli.load(pyproject)['project']

project   = str(pkg_meta['name'])
copyright = '2018-2026, PSYDAC Developers'
author    = str(pkg_meta['authors'][0]['name'])
release   = str(pkg_meta['version'])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
'sphinx.ext.inheritance_diagram',
'numpydoc',
'sphinx.ext.viewcode',
'sphinx.ext.graphviz',
'sphinx.ext.autodoc',
'sphinx.ext.autosummary',
'sphinx.ext.githubpages',
'sphinx_math_dollar',
'sphinx.ext.mathjax',
'nbsphinx',
'myst_parser',
]

from docutils.nodes import FixedTextElement, literal,math
from docutils.nodes import  comment, doctest_block, image, literal_block, math_block, paragraph, pending, raw, rubric, substitution_definition, target
math_dollar_node_blacklist = (literal,math,doctest_block, image, literal_block,  math_block,  pending,  raw,rubric, substitution_definition,target)

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

#numpydoc_class_members_toctree = False, nothing changed using this
numpydoc_show_class_members = False
templates_path = ['_templates']
exclude_patterns = []
add_module_names = False
#inheritance_graph_attrs = dict(rankdir="LR", size='"6.0, 4.0"',
#                               fontsize=14, ratio='compress')

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
#html_static_path = ['_static']

html_logo = "logo/psydac_square.svg"

html_theme_options = {
#    "repository_branch": "devel",
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pyccel/psydac",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
}   

# -- Options for myst_parser -------------------------------------------------
myst_heading_anchors = 3

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = 'bysource'

# inheritance diagrams
inheritance_graph_attrs = dict(rankdir="LR", ratio='auto',
                               fontsize="12")

inheritance_node_attrs = dict(shape='ellipse', fontsize="12", height=0.65,
                              color='maroon4', style='filled')
