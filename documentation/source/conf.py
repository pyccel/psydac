# pylint: disable=redefined-builtin

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

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

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
