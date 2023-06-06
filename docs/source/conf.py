# pylint: disable=redefined-builtin

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# add these directories to sys.path here.
import pathlib
import sys
import tomli

autodoc_mock_imports = ['sympy', 'sympde', 'numpy', 'scipy', 'mpi4py', 'pyccel', 'h5py', 'yaml', 'gelato', 'pyevtk', 'matplotlib']
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

with open('../../pyproject.toml', mode='rb') as pyproject:
    pkg_meta = tomli.load(pyproject)['project']

project   = str(pkg_meta['name'])
copyright = '2018-2023, Psydac Developers'
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

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
