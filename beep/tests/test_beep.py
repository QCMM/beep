"""
Unit and regression test for the beep package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import beep


def test_beep_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "beep" in sys.modules
