"""A binding energy evaluation platform and database for molecules on interstellar ice-grain mantels"""

# New structure imports
from . import core
from . import models

try:
    from . import adapters
except ImportError:
    pass

from importlib.metadata import version as _version

__version__ = _version("beep")
