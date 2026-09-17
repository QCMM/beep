"""A binding energy evaluation platform and database for molecules on interstellar ice-grain mantels"""

# New structure imports
from . import core
from . import models

try:
    from . import adapters
except ImportError:
    pass

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("beep")
except PackageNotFoundError:
    # Running from an uninstalled checkout (PYTHONPATH / editable copy
    # without metadata): fall back rather than failing at import time.
    __version__ = "0.0.0+unknown"
