'''
Date: 2024-11-28 17:46:44
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2024-11-28 22:52:57
FilePath: /MineStudio/docs/source/conf.py
'''
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = 'MineStudio'
copyright = '2024, CraftJarvis'
author = 'CraftJarvis'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.doctest',
   'sphinx.ext.intersphinx',
   'sphinx.ext.todo',
   'sphinx.ext.coverage',
   'sphinx.ext.mathjax',
   'sphinx.ext.ifconfig',
   'sphinx.ext.viewcode',
   'sphinx.ext.githubpages',
   'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

sys.path.insert(0, os.path.abspath('../minestudio'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_theme_options = {
  "show_nav_level": 2
}