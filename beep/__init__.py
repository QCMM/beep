"""A binding energy evaluation platform and database for molecules on interstellar ice-grain mantels"""

# New structure imports
from . import core
from . import models

try:
    from . import adapters
except ImportError:
    pass

__version__ = "0.2.0"
