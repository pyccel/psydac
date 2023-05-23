# pylint: disable=redefined-builtin

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from psydac import __version__

project = 'PSYDAC'
copyright = '2018-2023, Psydac Developers'
author = 'Psydac Developers'
release = __version__

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

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
