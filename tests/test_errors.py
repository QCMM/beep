"""Tests for beep/core/errors.py and beep/core/exceptions.py."""
import pytest

from beep.core.errors import DatasetNotFound, LevelOfTheoryNotFound
from beep.core.exceptions import (
    MoleculeNotSetError,
    OptimizationMethodNotSetError,
    DataNotLoadedError,
)


def test_dataset_not_found_raises():
    with pytest.raises(DatasetNotFound):
        raise DatasetNotFound("dataset xyz not found")


def test_level_of_theory_not_found_raises():
    with pytest.raises(LevelOfTheoryNotFound):
        raise LevelOfTheoryNotFound("B3LYP not available")


def test_molecule_not_set_default_message():
    err = MoleculeNotSetError()
    assert "set_molecule" in err.message


def test_optimization_method_not_set_default_message():
    err = OptimizationMethodNotSetError()
    assert "method" in err.message.lower() or "Method" in err.message


def test_data_not_loaded_default_message():
    err = DataNotLoadedError()
    assert "load_data" in err.message


def test_custom_message():
    err = MoleculeNotSetError("custom")
    assert err.message == "custom"
