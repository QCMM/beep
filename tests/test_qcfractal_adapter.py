"""Tests for beep/adapters/qcfractal_adapter.py — all server calls mocked."""
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from beep.adapters.qcfractal_adapter import (
    connect,
    get_or_create_opt_dataset,
    check_collection_exists,
    fetch_opt_molecules,
    check_for_completion,
)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@patch("beep.adapters.qcfractal_adapter.ptl")
def test_connect_calls_fractal_client(mock_ptl):
    mock_ptl.FractalClient.return_value = MagicMock()
    client = connect("localhost:7777")
    mock_ptl.FractalClient.assert_called_once_with(
        address="localhost:7777", verify=False,
        username=None, password=None,
    )


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def test_get_or_create_existing():
    mock_client = MagicMock()
    mock_ds = MagicMock()
    mock_client.get_collection.return_value = mock_ds
    result = get_or_create_opt_dataset(mock_client, "test_ds")
    assert result is mock_ds


@patch("beep.adapters.qcfractal_adapter.ptl")
def test_get_or_create_new(mock_ptl):
    mock_client = MagicMock()
    # First call raises KeyError, second call returns the new dataset
    mock_ds = MagicMock()
    mock_client.get_collection.side_effect = [KeyError("not found"), mock_ds]
    mock_ptl.collections.OptimizationDataset.return_value = MagicMock()

    result = get_or_create_opt_dataset(mock_client, "new_ds")
    assert result is mock_ds


def test_check_collection_exists_true():
    mock_client = MagicMock()
    mock_client.get_collection.return_value = MagicMock()
    assert check_collection_exists(mock_client, "OptimizationDataset", "ds") is True


def test_check_collection_exists_false():
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = KeyError("not found")
    assert check_collection_exists(mock_client, "OptimizationDataset", "ds") is False


# ---------------------------------------------------------------------------
# Molecule queries
# ---------------------------------------------------------------------------

def test_fetch_opt_molecules_filters_by_status():
    mock_ds = MagicMock()

    # Two records: one COMPLETE, one ERROR
    rec_complete = MagicMock()
    rec_complete.status = "COMPLETE"
    rec_complete.get_final_molecule.return_value = MagicMock()

    rec_error = MagicMock()
    rec_error.status = "ERROR"

    mock_ds.get_record.side_effect = [rec_complete, rec_error]

    result = fetch_opt_molecules(mock_ds, ["entry1", "entry2"], "opt_lot")
    assert len(result) == 1
    assert result[0][0] == "entry1"


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

def test_check_for_completion_all_done():
    mock_client = MagicMock()
    rec1 = MagicMock()
    rec1.status = "COMPLETE"
    rec2 = MagicMock()
    rec2.status = "COMPLETE"
    mock_client.query_procedures.return_value = [rec1, rec2]

    done, counts = check_for_completion(mock_client, ["pid1", "pid2"])
    assert done is True
    assert counts["COMPLETE"] == 2


def test_check_for_completion_incomplete():
    mock_client = MagicMock()
    rec1 = MagicMock()
    rec1.status = "COMPLETE"
    rec2 = MagicMock()
    rec2.status = "INCOMPLETE"
    mock_client.query_procedures.return_value = [rec1, rec2]

    done, counts = check_for_completion(mock_client, ["pid1", "pid2"])
    assert done is False
    assert counts["INCOMPLETE"] == 1
