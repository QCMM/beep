"""Tests for the ManybodyDataset entry helper in beep/workflows/mbe.py."""
import pytest

from beep.workflows.mbe import _add_entry


class _NoOverwriteDataset:
    """qcportal 0.64 ManybodyDataset.add_entry has no ``overwrite`` kwarg."""
    def __init__(self):
        self.calls = []

    def add_entry(self, name, initial_molecule):
        self.calls.append((name, initial_molecule))


class _OverwriteDataset:
    def __init__(self):
        self.calls = []

    def add_entry(self, name, initial_molecule, overwrite=False):
        self.calls.append((name, initial_molecule, overwrite))


class _ValidationFailingDataset:
    def add_entry(self, name, initial_molecule, overwrite=False):
        raise TypeError("initial_molecule must be a Molecule, got str")


def test_add_entry_falls_back_when_overwrite_unsupported():
    ds = _NoOverwriteDataset()
    _add_entry(ds, "W22_01", "mol", True, [])
    assert ds.calls == [("W22_01", "mol")]


def test_add_entry_uses_overwrite_when_supported():
    ds = _OverwriteDataset()
    _add_entry(ds, "W22_01", "mol", True, [])
    assert ds.calls == [("W22_01", "mol", True)]


def test_add_entry_reraises_unrelated_type_errors():
    """Only the ``overwrite`` signature mismatch is swallowed; a TypeError
    from molecule validation must propagate."""
    with pytest.raises(TypeError, match="must be a Molecule"):
        _add_entry(_ValidationFailingDataset(), "W22_01", "mol", True, [])


def test_add_entry_skips_existing_when_not_updating():
    ds = _OverwriteDataset()
    _add_entry(ds, "W22_01", "mol", False, ["W22_01"])
    assert ds.calls == []
