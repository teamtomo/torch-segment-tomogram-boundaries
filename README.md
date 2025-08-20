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

## TODO

1. Fix p02 script to use tomo dims from constants for colume resizing
2. Overhall the design to be API friendly and not script like as it is currently.
3. Remove redundant losses from the architecture, weighted_bce +/- dice is sufficient.
4. Move network arch to constants and import them when defining in relevant modules
5. Consider vol_utils.py and im_utils.py modules for storing commonly used functions and prevent code duplication
