# Installation

spatialbatch needs Python 3.12 or newer.
We recommend [uv][], which installs Python for you and keeps each project's dependencies in their own virtual environment.

## PyPI

Install the latest release from [PyPI][]:

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv add spatialbatch
```

::::

::::{tab-item} Pip
:sync: pip

```bash
pip install spatialbatch
```

::::
:::::

The default install includes spatialdata's `extra` ([spatialdata-plot][], [spatialdata-io][] and [napari-spatialdata][]), so plotting and the technology readers work out of the box.

To get a pre-release (e.g. a release candidate) once a final release exists, opt in explicitly:

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv add --prerelease allow spatialbatch
```

::::

::::{tab-item} Pip
:sync: pip

```bash
pip install --pre spatialbatch
```

::::
:::::

## Development version

Install the latest development version from GitHub:

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv add git+https://github.com/rushin682/spatialbatch.git
```

::::

::::{tab-item} Pip
:sync: pip

```bash
pip install git+https://github.com/rushin682/spatialbatch.git
```

::::
:::::

To work on spatialbatch itself, see {doc}`contributing`.

[uv]: https://docs.astral.sh/uv/
[pypi]: https://pypi.org/project/spatialbatch
[spatialdata-plot]: https://spatialdata.scverse.org/projects/plot/en/latest/
[spatialdata-io]: https://spatialdata.scverse.org/projects/io/en/latest/
[napari-spatialdata]: https://spatialdata.scverse.org/projects/napari/en/latest/
