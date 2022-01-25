"""A binding energy evaluation platforma and database for molecules on interstellar ice-grain mantels"""

# Add imports here
from .beep import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
