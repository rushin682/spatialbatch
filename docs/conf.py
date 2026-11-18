# Configuration file for the Sphinx documentation builder.
# `scverse_doc` sets the theme and shared defaults, see https://scverse-doc.readthedocs.io
# For the full list of options, see https://www.sphinx-doc.org/page/usage/configuration.html

from datetime import datetime
from importlib.metadata import metadata

from scverse_doc.registry import intersphinx

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("spatialbatch")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = release = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "rushin682",
    "github_repo": project,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "scverse_doc",
    "scverse_misc.sphinx_ext",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
]

bibtex_bibfiles = ["references.bib"]
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

xref_overrides = {
    # If building the documentation fails because a type annotation links to a private path
    # (e.g. `igraph._igraph.GraphBase`), map it to the documented name here.
    # For a missing link that can’t be fixed this way, add it to `nitpick_ignore` instead.
    #     "igraph._igraph.GraphBase": "igraph.GraphBase",
}

# Packages outside the scverse registry that the docstrings link to.
intersphinx_mapping = intersphinx() | {
    "annbatch": ("https://annbatch.readthedocs.io/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "zarrs": ("https://zarrs-python.readthedocs.io/en/latest/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
}

# `BackingArray` is a private type variable in annbatch's signatures, with nothing to link to.
nitpick_ignore = [("py:class", "BackingArray")]


def _resolve_typeddict(app, env, node, contnode):
    """Resolve a failed ``:class:`` reference against intersphinx ``py:typeddict`` entries.

    annbatch's inventory lists its TypedDicts (e.g. `annbatch.types.LoadRequest`) as ``py:typeddict``,
    an object type this build's Python domain doesn't know, so ``:class:`` references to them
    (including those in docstrings inherited from annbatch) find nothing.
    """
    from docutils import nodes

    if node.get("refdomain") != "py" or node.get("reftype") != "class":
        return None
    entry = env.intersphinx_inventory.get("py:typeddict", {}).get(node["reftarget"])
    if entry is None:
        return None
    ref = nodes.reference("", "", internal=False, refuri=entry.uri, reftitle=entry.project_name)
    ref.append(contnode)
    return ref


def setup(app):
    """Register the typeddict reference fallback."""
    app.connect("missing-reference", _resolve_typeddict)
