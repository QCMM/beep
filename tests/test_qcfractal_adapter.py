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
    check_jobs_status,
    create_keyword_set,
    get_job_ids,
    get_zpve_mol,
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

def test_get_job_ids_skips_missing_entries():
    """get_job_ids must not raise if entry_list contains names that aren't
    in the dataset — pose generation can short-return, leaving requested
    names un-inserted (see reports/bugs/beep_sampling_get_job_ids_missing_entries.md).
    """
    mock_ds = MagicMock()
    mock_ds.entry_names = ["e1", "e2"]  # "e3" is asked for but absent

    rec1 = MagicMock(); rec1.id = 101
    rec2 = MagicMock(); rec2.id = 102
    mock_ds.get_record.side_effect = lambda n, lot: {"e1": rec1, "e2": rec2}[n]

    pids = get_job_ids(mock_ds, ["e1", "e3", "e2"], "opt_lot")

    assert pids == [101, 102]
    # get_record never called for the missing name — saved a server round-trip
    assert mock_ds.get_record.call_count == 2
    called_names = [c.args[0] for c in mock_ds.get_record.call_args_list]
    assert "e3" not in called_names


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


def _service_record(rec_id, status, is_service=True):
    r = MagicMock()
    r.id = rec_id
    r.status = status
    r.is_service = is_service
    return r


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_auto_recovers_errored_services():
    """When a parent ReactionRecord is at ERROR, check_jobs_status should
    reset it once so the server re-iterates the service. If the children
    have since been fixed, the service transitions to COMPLETE on its own.
    """
    mock_client = MagicMock()
    logger = MagicMock()

    err_svc = _service_record(101, RecordStatusEnum.error, is_service=True)
    done_svc = _service_record(101, RecordStatusEnum.complete, is_service=True)

    # First poll: service is ERROR → should trigger reset.
    # Second poll: server has re-iterated and the service is now COMPLETE → exit.
    mock_client.get_records.side_effect = [[err_svc], [done_svc]]

    check_jobs_status(mock_client, [101], logger, wait_interval=1)

    mock_client.reset_records.assert_called_once_with([101])


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_does_not_reset_leaf_records():
    """Singlepoint / leaf records (is_service=False) must NEVER be
    auto-reset — that would trigger needless recomputation."""
    mock_client = MagicMock()
    logger = MagicMock()

    err_leaf = _service_record(202, RecordStatusEnum.error, is_service=False)
    # Loop exits after one poll because INCOMPLETE==0 and no recoverables.
    mock_client.get_records.return_value = [err_leaf]

    check_jobs_status(mock_client, [202], logger, wait_interval=1)

    mock_client.reset_records.assert_not_called()


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_resets_each_service_only_once():
    """A service stuck at ERROR (real child error) must not be reset on
    every polling cycle. After the first reset, subsequent ERRORs are
    accepted as terminal."""
    mock_client = MagicMock()
    logger = MagicMock()

    err_svc_a = _service_record(303, RecordStatusEnum.error, is_service=True)
    err_svc_b = _service_record(303, RecordStatusEnum.error, is_service=True)
    # Three polls in a row, same service in ERROR; only the first triggers reset.
    mock_client.get_records.side_effect = [[err_svc_a], [err_svc_b], [err_svc_b]]

    check_jobs_status(mock_client, [303], logger, wait_interval=1, max_wait=2)

    # After the second poll, services_to_recover is empty and
    # INCOMPLETE==0, so the loop exits there. Only one reset call.
    assert mock_client.reset_records.call_count == 1
    mock_client.reset_records.assert_called_with([303])


def test_get_zpve_mol_filters_incomplete_records():
    """get_zpve_mol must skip hessian records whose `properties` is None —
    a 'complete'-tagged record can still be mid-write or otherwise
    inconsistent, and `result.return_result` would crash on it.

    Regression for the case where multiple hessian records exist for the
    same molecule + spec and the first one is unusable.
    """
    mock_client = MagicMock()

    # Mock molecule with multiple symbols so we skip the atom early-return.
    mock_mol = MagicMock()
    mock_mol.symbols = ["O", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H"]
    mock_mol.dict.return_value = {"identifiers": {"molecular_formula": "H10O"}}
    mock_client.get_molecules.return_value = [mock_mol]

    # Bad record: complete-tagged but properties is None (server inconsistency).
    bad_record = MagicMock()
    bad_record.properties = None

    mock_client.query_singlepoints.return_value = iter([bad_record])

    # With our filter, this should hit the "no usable record" branch and
    # return (None, True) instead of crashing on result.return_result.
    zpve, ok = get_zpve_mol(mock_client, 12345, "mpwb1k-d3bj_def2-tzvpd")
    assert zpve is None
    assert ok is True

    # Verify the query was made with status=complete, narrowing to
    # server-side completes before we even client-side filter.
    kwargs = mock_client.query_singlepoints.call_args.kwargs
    assert kwargs["status"] == RecordStatusEnum.complete


@patch("beep.adapters.qcfractal_adapter.time.sleep", lambda *a, **kw: None)
def test_check_jobs_status_auto_recover_disabled():
    """Setting auto_recover_services=False preserves the old behavior:
    ERROR is fully terminal, no reset attempts."""
    mock_client = MagicMock()
    logger = MagicMock()

    err_svc = _service_record(404, RecordStatusEnum.error, is_service=True)
    mock_client.get_records.return_value = [err_svc]

    check_jobs_status(
        mock_client, [404], logger, wait_interval=1,
        auto_recover_services=False,
    )

    mock_client.reset_records.assert_not_called()


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


# ---------------------------------------------------------------------------
# fetch_reaction_values — separated-pair recombination, integrated-spec skip
# ---------------------------------------------------------------------------

def _fake_reaction_dataset(records_by_spec):
    """Build a mock ReactionDataset for fetch_reaction_values tests.

    ``records_by_spec`` is a dict: spec_name → {entry_name: total_energy_hartree}.
    """
    ds = MagicMock()
    ds.specification_names = list(records_by_spec.keys())

    def _iterate(specification_names, status):
        sname = specification_names[0]
        for entry_name, energy in records_by_spec[sname].items():
            rec = MagicMock()
            rec.total_energy = energy
            yield entry_name, sname, rec

    ds.iterate_records.side_effect = _iterate
    return ds


@patch("beep.adapters.qcfractal_adapter._stoich_dataset_name",
       return_value="be_H2O_W5_01_bsse")
def test_fetch_reaction_values_d4_separated_pair(mock_ds_name):
    """Separated pair for D4 recombines into composite column."""
    from beep.adapters.qcfractal_adapter import fetch_reaction_values

    # In hartree; composite column should be sum of the two pieces
    records = {
        "b3lyp_def2-tzvp": {"entry1": -1.0},
        "b3lyp-d4":         {"entry1": -0.01},
    }
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = _fake_reaction_dataset(records)

    df = fetch_reaction_values(mock_client, "be_H2O_W5_01", stoich="bsse")

    assert "b3lyp-d4/def2-tzvp" in df.columns
    # Sum in kcal/mol (hartree2kcalmol ≈ 627.5)
    import qcelemental as qcel
    expected = (-1.0 - 0.01) * qcel.constants.hartree2kcalmol
    assert df.loc["entry1", "b3lyp-d4/def2-tzvp"] == pytest.approx(expected)


@patch("beep.adapters.qcfractal_adapter._stoich_dataset_name",
       return_value="be_H2O_W5_01_bsse")
def test_fetch_reaction_values_skips_integrated_spec(mock_ds_name, caplog):
    """Integrated specs (dispersion suffix before underscore) are skipped with a warning."""
    from beep.adapters.qcfractal_adapter import fetch_reaction_values

    records = {
        "pbe_def2-tzvp":           {"entry1": -1.0},
        "pbe-d3bj":                {"entry1": -0.02},
        "pbe-d3bj_def2-tzvp":      {"entry1": -99.0},   # integrated — should be skipped
    }
    mock_client = MagicMock()
    mock_client.get_dataset.return_value = _fake_reaction_dataset(records)

    # The "beep" logger may have propagate disabled from prior tests — force
    # propagation so caplog can see the warning.
    import logging
    beep_logger = logging.getLogger("beep")
    prev_propagate = beep_logger.propagate
    beep_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="beep"):
            df = fetch_reaction_values(mock_client, "be_H2O_W5_01", stoich="bsse")
    finally:
        beep_logger.propagate = prev_propagate

    assert any("integrated dispersion spec 'pbe-d3bj_def2-tzvp'" in r.message
               for r in caplog.records)

    # Composite column comes from the separated pair, not the skipped integrated value
    import qcelemental as qcel
    expected = (-1.0 - 0.02) * qcel.constants.hartree2kcalmol
    assert df.loc["entry1", "pbe-d3bj/def2-tzvp"] == pytest.approx(expected)
