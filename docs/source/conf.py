# -*- coding: utf-8 -*-
import sys
import os
import datetime
import sphinx_gallery
from sphinx_gallery.sorting import ExampleTitleSortKey
from pyproximal import __version__

# Sphinx needs to be able to import the package to use autodoc and get the version number
sys.path.insert(0, os.path.abspath('../../pyproximal'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    "sphinx.ext.intersphinx",
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    'nbsphinx',
    'sphinx_gallery.gen_gallery',
    'sphinxemoji.sphinxemoji',
]

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("http://scikit-learn.org/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pyfftw": ("https://pyfftw.readthedocs.io/en/latest/", None),
    "pylops": ("https://pylops.readthedocs.io/en/latest/", None),
}

## Generate autodoc stubs with summaries from code
autosummary_generate = True

## Include Python objects as they appear in source files
autodoc_member_order = 'bysource'

## Default flags used by autodoc directives
autodoc_default_flags = ['members']

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': ['../../examples', '../../tutorials',],
    # path where to save gallery generated examples
    'gallery_dirs': ['gallery', 'tutorials'],
    'filename_pattern': '\.py',
    # Remove the "Download all examples" button from the top level gallery
    'download_all_examples': False,
    # Sort gallery example by file name instead of number of lines (default)
    'within_subsection_order': ExampleTitleSortKey,
    # directory where function granular galleries are stored
    'backreferences_dir': 'api/generated/backreferences',
    # Modules for which function level galleries are created.
    'doc_module': 'pyproximal',
    # Insert links to documentation of objects in the examples
    'reference_url': {'pyproximal': None},
    # Allow animations
    'matplotlib_animations': True,
}

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ['png']

# Sphinx project configuration
templates_path = ['_templates']
exclude_patterns = ["_build", "**.ipynb_checkpoints", "**.ipynb", "**.md5"]
source_suffix = '.rst'

# The encoding of source files.
source_encoding = 'utf-8-sig'
master_doc = 'index'

# General information about the project
year = datetime.date.today().year
project = 'PyProximal'
copyright = '{}, Matteo Ravasi'.format(year)

# Version
version = __version__
if len(version.split('+')) > 1 or version == 'unknown':
    version = 'dev'

# These enable substitutions using |variable| in the rst files
rst_epilog = """
.. |year| replace:: {year}
""".format(year=year)

html_static_path = ["_static"]
html_last_updated_fmt = '%b %d, %Y'
html_title = 'PyProximal'
html_short_title = 'PyProximal'
html_logo = '_static/pyproximal.png'
html_favicon = 'favicon.ico'
html_extra_path = []
pygments_style = 'default'
add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# Theme config
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    "logo": {
            "image_light": "pyproximal_b.png",
            "image_dark": "pyproximal.png",
    }
}
html_context = {
    'menu_links_name': 'Repository',
    'menu_links': [
        ('<i class="fa fa-github fa-fw"></i> Source Code', 'https://github.com/PyLops/pyproximal'),
        ('<i class="fa fa-users fa-fw"></i> Contributing', 'https://github.com/PyLops/pyproximal/blob/master/CONTRIBUTING.md'),
    ],
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    'doc_path': 'docs/source',
    'galleries': sphinx_gallery_conf['gallery_dirs'],
    'gallery_dir': dict(zip(sphinx_gallery_conf['gallery_dirs'],
                            sphinx_gallery_conf['examples_dirs'])),
    'github_project': 'PyLops',
    'github_repo': 'pyproximal',
    'github_version': 'master',
}

# Math macros
mathjax3_config = {
    "tex": {
        "macros": {
            "prox": r"\operatorname{prox}",
            "sgn": r"\operatorname{sgn}",
            "argmin": r"\operatorname*{argmin}",
            "diag": r"\operatorname{diag}",
            "rank": r"\operatorname{rank}",
        }
    }
}

# Load the custom CSS files (needs sphinx >= 1.6 for this to work)
def setup(app):
    app.add_css_file("style.css")
