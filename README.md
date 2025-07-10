# torch-tomo-slab

[![License](https://img.shields.io/pypi/l/torch-tomo-slab.svg?color=green)](https://github.com/shahpnmlab/torch-tomo-slab/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-tomo-slab.svg?color=green)](https://pypi.org/project/torch-tomo-slab)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-tomo-slab.svg?color=green)](https://python.org)
[![CI](https://github.com/shahpnmlab/torch-tomo-slab/actions/workflows/ci.yml/badge.svg)](https://github.com/shahpnmlab/torch-tomo-slab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/shahpnmlab/torch-tomo-slab/branch/main/graph/badge.svg)](https://codecov.io/gh/shahpnmlab/torch-tomo-slab)

A simple Unet application to detect boundaries of tomographic volumes

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork shahpnmlab/torch-tomo-slab --clone
# or just
# gh repo clone shahpnmlab/torch-tomo-slab
cd torch-tomo-slab
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
