"""A binding energy evaluation platform and database for molecules on interstellar ice-grain mantels"""

# New structure imports
from . import core
from . import models

try:
    from . import adapters
except ImportError:
    pass

# Handle versioneer
try:
    from ._version import get_versions
    versions = get_versions()
    __version__ = versions["version"]
    __git_revision__ = versions["full-revisionid"]
    del get_versions, versions
except Exception:
    __version__ = "0.2.0"
    __git_revision__ = ""
