# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('./multivariate'))

# -- Project information -----------------------------------------------------

project = 'package "data-depth"'
copyright = '2022, Pavlo Mozharovskyi, Rainer Dyckerhoff, Oleksii Pokotylo, Romain Valla'
author = 'Pavlo Mozharovskyi, Rainer Dyckerhoff, Oleksii Pokotylo, Romain Valla'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.doctest',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx_copybutton',
              'jupyter_sphinx',
              'python_docs_theme',
              'numpydoc',
              'nbsphinx',
]

# Remove '>>> ' from the copy button and
# set only lines with prompts are copied

copybutton_prompt_text = ">>> "
copybutton_only_copy_prompt_lines = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_logo = '_static/depth-logo.jpg'
html_favicon = '_static/depth-logo.jpg'
html_theme_options = {
    "github_url": "https://github.com/data-depth"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

add_module_names = False
