# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PreCosyne BrainHack 2026"
copyright = "2026, Pynapple team"
author = "Pynapple team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "myst_nb",
    "sphinx_copybutton",
    # "sphinx_togglebutton",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

master_doc = "index"
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nb_execution_timeout = -1
nb_execution_raise_on_error = True
myst_enable_extensions = ["colon_fence", "dollarmath", "attrs_inline"]
nb_execution_mode = "cache"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["mycss.css"]
# html_js_files = ["myjs.js"]

html_theme_options = {
    "logo": {
        "text": [],
    },
    "secondary_sidebar_items": {
        "**": ["page-toc"],
        "index": [],
    },
    "navbar_persistent": [],
}
