'''
Date: 2024-11-28 17:46:44
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2024-11-28 23:13:19
FilePath: /MineStudio/docs/source/conf.py
'''
import os
import sys
from datetime import datetime

project = 'MineStudio'
copyright = str(datetime.now().year) + ", The CraftJarvis Team"
author = 'The CraftJarvis Team'
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

html_title = f"MineStudio {release}"