# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# add these directories to sys.path here.
import pathlib
import sys
autodoc_mock_imports = ['numpy', 'scipy', 'sympy', 'sympde', 'mpi4py', 'pyccel', 'h5py', 'yaml']
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

project = 'PSYDAC'
copyright = '2023, Numerical Methods in Plasma Physics division, Max Planck Institute for Plasma Physics Garching'
author = 'Numerical Methods in Plasma Physics division, Max Planck Institute for Plasma Physics Garching'
release = 'v0.1'

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
]

#numpydoc_class_members_toctree = False, nothing changed using this
numpydoc_show_class_members = False
templates_path = ['_templates']
exclude_patterns = []
add_module_names = False
#inheritance_graph_attrs = dict(rankdir="LR", size='"6.0, 4.0"',
#                               fontsize=14, ratio='compress')


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme' # used to be 'piccolo_theme'
html_static_path = ['_static']
