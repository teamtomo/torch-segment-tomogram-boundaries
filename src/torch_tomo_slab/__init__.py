"""A simple Unet application to detect boundaries of tomographic volumes"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tomo-slab")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Pranav NM Shah"
__email__ = "p.shah.lab@gmail.com"
