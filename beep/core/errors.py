"""
Custom Error Module

This module defines custom error classes for handling specific error conditions in your application.

- DatasetNotFound: Raised when a dataset or collection is not found.
- ConfigurationError: Raised when there is a problem with the application's configuration.
"""

class DatasetNotFound(Exception):
    """
    Raised when a dataset or collection is not found.
    """
    pass

class LevelOfTheoryNotFound(Exception):
    """
    Raised when the level of theory is not found.
    """
    pass

