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


# ---------------------------------------------------------------------------
# Dispersion splitting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, expected",
    [
        ("pbe-d3bj",    ("pbe",    "pbe-d3bj",    "dftd3")),
        ("b3lyp-d3mbj", ("b3lyp",  "b3lyp-d3mbj", "dftd3")),
        ("mpwb1k-d3m",  ("mpwb1k", "mpwb1k-d3m",  "dftd3")),
        ("pbe-d3",      ("pbe",    "pbe-d3",      "dftd3")),
        ("b3lyp-d4",    ("b3lyp",  "b3lyp-d4",    "dftd4")),
        ("PBE-D3BJ",    ("PBE",    "PBE-D3BJ",    "dftd3")),  # case preserved in bare/full
        ("wb97x-v",     ("wb97x-v", None, None)),  # intrinsic dispersion, not split
        ("wb97m-v",     ("wb97m-v", None, None)),
        ("hf3c",        ("hf3c",   None, None)),
        ("pbe",         ("pbe",    None, None)),
    ],
)
def test_split_dispersion(method, expected):
    from beep.adapters.qcfractal_adapter import _split_dispersion
    assert _split_dispersion(method) == expected


# ---------------------------------------------------------------------------
# compute_be_dft_energies — dispatch to separated-pair vs integrated form
# ---------------------------------------------------------------------------

def _fake_submit_result(n_inserted=1, n_existing=0):
    r = MagicMock()
    r.n_inserted = n_inserted
    r.n_existing = n_existing
    return r


@patch("beep.adapters.qcfractal_adapter.submit_energies")
def test_compute_be_dft_energies_splits_dispersion(mock_submit):
    """D3BJ method → two submit_energies calls per stoich: bare DFT + bare dispersion."""
    from beep.adapters.qcfractal_adapter import compute_be_dft_energies, STOICH_TYPES

    mock_submit.return_value = _fake_submit_result()
    logger = MagicMock()

    compute_be_dft_energies(
        client=MagicMock(), rdset_base_name="be_H2O_W5_01",
        all_dft=["pbe-d3bj_def2-tzvp"], tag="test",
        program="psi4", logger=logger,
    )

    # Two calls per stoich type: bare DFT (psi4) + bare dispersion (dftd3)
    assert mock_submit.call_count == 2 * len(STOICH_TYPES)

    dft_calls = [c for c in mock_submit.call_args_list if c.kwargs["program"] == "psi4"]
    disp_calls = [c for c in mock_submit.call_args_list if c.kwargs["program"] == "dftd3"]
    assert len(dft_calls) == len(STOICH_TYPES)
    assert len(disp_calls) == len(STOICH_TYPES)

    assert dft_calls[0].kwargs["method"] == "pbe"
    assert dft_calls[0].kwargs["basis"] == "def2-tzvp"
    assert disp_calls[0].kwargs["method"] == "pbe-d3bj"
    assert disp_calls[0].kwargs["basis"] is None


@patch("beep.adapters.qcfractal_adapter.submit_energies")
def test_compute_be_dft_energies_integrated_for_non_dispersion(mock_submit):
    """HF-3c has no dispersion suffix → single integrated spec per stoich."""
    from beep.adapters.qcfractal_adapter import compute_be_dft_energies, STOICH_TYPES

    mock_submit.return_value = _fake_submit_result()
    logger = MagicMock()

    compute_be_dft_energies(
        client=MagicMock(), rdset_base_name="be_H2O_W5_01",
        all_dft=["hf3c_minix"], tag="test",
        program="psi4", logger=logger,
    )

    assert mock_submit.call_count == len(STOICH_TYPES)
    for call in mock_submit.call_args_list:
        assert call.kwargs["method"] == "hf3c"
        assert call.kwargs["basis"] == "minix"
        assert call.kwargs["program"] == "psi4"


@patch("beep.adapters.qcfractal_adapter.submit_energies")
def test_compute_be_dft_energies_d4_uses_dftd4_program(mock_submit):
    """D4 dispersion routes to dftd4 program (not dftd3)."""
    from beep.adapters.qcfractal_adapter import compute_be_dft_energies

    mock_submit.return_value = _fake_submit_result()

    compute_be_dft_energies(
        client=MagicMock(), rdset_base_name="be_H2O_W5_01",
        all_dft=["b3lyp-d4_def2-tzvp"], tag="test",
        program="psi4", logger=MagicMock(),
    )

    disp_calls = [c for c in mock_submit.call_args_list if c.kwargs["program"] == "dftd4"]
    assert len(disp_calls) > 0
    assert disp_calls[0].kwargs["method"] == "b3lyp-d4"
    assert disp_calls[0].kwargs["basis"] is None
