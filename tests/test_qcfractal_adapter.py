"""Tests for beep/adapters/qcfractal_adapter.py — all server calls mocked."""
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from qcportal.record_models import RecordStatusEnum

from beep.adapters.qcfractal_adapter import (
    connect,
    get_or_create_opt_dataset,
    check_collection_exists,
    fetch_opt_molecules,
    check_for_completion,
    create_keyword_set,
    query_keywords,
    is_complete,
    is_incomplete,
    is_error,
    status_label,
)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@patch("beep.adapters.qcfractal_adapter.PortalClient")
def test_connect_calls_portal_client(mock_portal):
    mock_portal.return_value = MagicMock()
    client = connect("localhost:7777")
    mock_portal.assert_called_once_with(
        address="localhost:7777", verify=False,
        username=None, password=None,
        show_motd=False,
    )


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def test_get_or_create_existing():
    mock_client = MagicMock()
    mock_ds = MagicMock()
    mock_client.get_dataset.return_value = mock_ds
    result = get_or_create_opt_dataset(mock_client, "test_ds")
    assert result is mock_ds


def test_get_or_create_new():
    mock_client = MagicMock()
    mock_ds = MagicMock()
    mock_client.get_dataset.side_effect = KeyError("not found")
    mock_client.add_dataset.return_value = mock_ds

    result = get_or_create_opt_dataset(mock_client, "new_ds")
    mock_client.add_dataset.assert_called_once_with("optimization", "new_ds")
    assert result is mock_ds


def test_check_collection_exists_true():
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = MagicMock()
    assert check_collection_exists(mock_client, "OptimizationDataset", "ds") is True


def test_check_collection_exists_false():
    mock_client = MagicMock()
    mock_client.get_dataset.side_effect = KeyError("not found")
    assert check_collection_exists(mock_client, "OptimizationDataset", "ds") is False


# ---------------------------------------------------------------------------
# Molecule queries
# ---------------------------------------------------------------------------

def test_fetch_opt_molecules_filters_by_status():
    mock_ds = MagicMock()
    sentinel_mol = MagicMock()

    # Two records: one complete, one error
    rec_complete = MagicMock()
    rec_complete.status = RecordStatusEnum.complete
    rec_complete.final_molecule = sentinel_mol

    rec_error = MagicMock()
    rec_error.status = RecordStatusEnum.error

    mock_ds.get_record.side_effect = [rec_complete, rec_error]

    result = fetch_opt_molecules(mock_ds, ["entry1", "entry2"], "opt_lot")
    assert len(result) == 1
    assert result[0][0] == "entry1"
    assert result[0][1] is sentinel_mol


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

def test_check_for_completion_all_done():
    mock_client = MagicMock()
    rec1 = MagicMock()
    rec1.status = RecordStatusEnum.complete
    rec2 = MagicMock()
    rec2.status = RecordStatusEnum.complete
    mock_client.get_records.return_value = [rec1, rec2]

    done, counts = check_for_completion(mock_client, [1, 2])
    assert done is True
    assert counts["COMPLETE"] == 2


def test_check_for_completion_incomplete():
    mock_client = MagicMock()
    rec1 = MagicMock()
    rec1.status = RecordStatusEnum.complete
    rec2 = MagicMock()
    rec2.status = RecordStatusEnum.running
    mock_client.get_records.return_value = [rec1, rec2]

    done, counts = check_for_completion(mock_client, [1, 2])
    assert done is False
    assert counts["INCOMPLETE"] == 1


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def test_status_helpers():
    assert is_complete(RecordStatusEnum.complete) is True
    assert is_complete(RecordStatusEnum.error) is False
    assert is_incomplete(RecordStatusEnum.running) is True
    assert is_incomplete(RecordStatusEnum.waiting) is True
    assert is_incomplete(RecordStatusEnum.complete) is False
    assert is_error(RecordStatusEnum.error) is True
    assert is_error(RecordStatusEnum.complete) is False


def test_status_label_mapping():
    assert status_label(RecordStatusEnum.complete) == "COMPLETE"
    assert status_label(RecordStatusEnum.running) == "INCOMPLETE"
    assert status_label(RecordStatusEnum.waiting) == "INCOMPLETE"
    assert status_label(RecordStatusEnum.error) == "ERROR"


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

def test_reexports_are_importable():
    """Adapter must re-export QCPortal types so workflows never import
    qcportal directly."""
    from beep.adapters.qcfractal_adapter import (
        FractalClient,
        PortalClient,
        Dataset,
        OptimizationDataset,
        ReactionDataset,
        Molecule,
    )
    assert FractalClient is PortalClient
    for cls in (FractalClient, Dataset, OptimizationDataset, ReactionDataset, Molecule):
        assert isinstance(cls, type), f"{cls} is not a class"


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

def test_create_keyword_set_returns_dict():
    result = create_keyword_set({"reference": "uks"})
    assert result == {"reference": "uks"}
    assert isinstance(result, dict)


def test_query_keywords_raises():
    mock_client = MagicMock()
    with pytest.raises(NotImplementedError):
        query_keywords(mock_client)


# ---------------------------------------------------------------------------
# Atom handling
# ---------------------------------------------------------------------------

def test_fetch_atom_molecule():
    from beep.adapters.qcfractal_adapter import fetch_atom_molecule

    mock_client = MagicMock()
    mock_ds = MagicMock()
    sentinel_mol = MagicMock()
    mock_entry = MagicMock()
    mock_entry.molecule = sentinel_mol

    mock_client.get_dataset.return_value = mock_ds
    mock_ds.get_entry.return_value = mock_entry

    result = fetch_atom_molecule(mock_client, "atoms", "O")
    mock_client.get_dataset.assert_called_once_with("singlepoint", "atoms")
    mock_ds.get_entry.assert_called_once_with("O")
    assert result is sentinel_mol


def test_fetch_atom_molecule_not_found():
    from beep.adapters.qcfractal_adapter import fetch_atom_molecule

    mock_client = MagicMock()
    mock_ds = MagicMock()
    mock_client.get_dataset.return_value = mock_ds
    mock_ds.get_entry.return_value = None

    with pytest.raises(KeyError, match="not found"):
        fetch_atom_molecule(mock_client, "atoms", "Xe")
