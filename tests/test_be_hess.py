"""Tests for beep/workflows/be_hess.py (offline, adapter fully mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from beep.workflows import be_hess


def _config(molecule="CO"):
    cfg = MagicMock()
    cfg.molecule = molecule
    cfg.level_of_theory = ["wb97x-v_def2-tzvp"]
    cfg.mace_models = []
    cfg.mace_dispersion = None
    cfg.program = "psi4"
    cfg.qc_keywords = None
    cfg.energy_tag = None
    return cfg


def _ready_dataset(name):
    ds = MagicMock()
    ds.name = name
    return ds


@pytest.mark.parametrize("cluster_name, expected_rdset", [
    ("cd5",    "be_CO_CD5_HF3C_MINIX"),      # no underscore: old parser gave "CO_cd5"
    ("W22_02", "be_CO_W22_02_HF3C_MINIX"),   # one underscore: the only case the old parser got right
    ("W12_1_b", "be_CO_W12_1_B_HF3C_MINIX"), # two underscores
])
def test_process_be_computation_uses_threaded_cluster_name(cluster_name, expected_rdset):
    """Regression: the cluster name was re-derived from an entry name via
    split('_')[-3:-1], which is only right for names with exactly one
    underscore. It is now carried alongside the dataset from
    check_refinement_status and used verbatim."""
    cfg = _config()
    ds_opt = _ready_dataset(f"CO_{cluster_name}")
    opt_stru = {f"CO_{cluster_name}_0001": MagicMock(), f"CO_{cluster_name}_0002": MagicMock()}
    logger = MagicMock()

    with patch.object(be_hess.qcf, "rmsd_filter_from_dataset", return_value=opt_stru), \
         patch.object(be_hess.qcf, "fetch_final_molecule", return_value="cluster-mol") as m_fetch, \
         patch.object(be_hess.qcf, "create_or_load_reaction_dataset", return_value="rdset") as m_create, \
         patch.object(be_hess.qcf, "compute_be_dft_energies", return_value=[1, 2]) as m_dft, \
         patch.object(be_hess.qcf, "compute_be_mace_energies", return_value=[]):
        ids = be_hess.process_be_computation(
            MagicMock(), logger, [(ds_opt, cluster_name)], "surf_ds",
            "smol", "hf3c_minix", 1, cfg,
        )

    assert ids == [1, 2]
    # cluster name used verbatim for the surface lookup ...
    assert m_fetch.call_args.args[1] == cluster_name
    # ... and for the reaction dataset name
    assert m_create.call_args.args[1] == expected_rdset
    m_dft.assert_called_once()


def test_process_be_computation_skips_cluster_with_empty_rmsd_result():
    """Regression: list(opt_stru.keys())[0] raised IndexError when the RMSD
    filter left no structures. Such a cluster is now skipped with a warning."""
    cfg = _config()
    ds_opt = _ready_dataset("CO_cd5")
    logger = MagicMock()

    with patch.object(be_hess.qcf, "rmsd_filter_from_dataset", return_value={}), \
         patch.object(be_hess.qcf, "fetch_final_molecule") as m_fetch, \
         patch.object(be_hess.qcf, "create_or_load_reaction_dataset") as m_create, \
         patch.object(be_hess.qcf, "compute_be_dft_energies") as m_dft:
        ids = be_hess.process_be_computation(
            MagicMock(), logger, [(ds_opt, "cd5")], "surf_ds",
            "smol", "hf3c_minix", 1, cfg,
        )

    assert ids == []
    m_fetch.assert_not_called()
    m_create.assert_not_called()
    m_dft.assert_not_called()
    assert logger.warning.called
    assert "cd5" in logger.warning.call_args.args[0]


def test_check_refinement_status_returns_dataset_cluster_pairs():
    """check_refinement_status pairs every ready dataset with its cluster name
    so the name never has to be parsed from an entry name downstream."""
    surf_ds = MagicMock()
    surf_ds.entry_names = ["cd5", "W22_02", "empty"]

    def fake_get_collection(client, kind, name):
        ds = MagicMock()
        ds.name = name
        if name == "CO_empty":
            ds.entry_names = []
        else:
            ds.entry_names = [f"{name}_0001", f"{name}_0002"]
        rec = MagicMock()
        rec.status = "complete"
        ds.get_record.return_value = rec
        return ds

    with patch.object(be_hess.qcf, "get_collection", side_effect=fake_get_collection), \
         patch.object(be_hess, "is_complete", return_value=True), \
         patch.object(be_hess, "is_incomplete", return_value=False), \
         patch.object(be_hess, "is_error", return_value=False):
        ready, counts = be_hess.check_refinement_status(
            MagicMock(), surf_ds, "CO", "hf3c_minix")

    assert [(ds.name, cn) for ds, cn in ready] == [("CO_cd5", "cd5"), ("CO_W22_02", "W22_02")]
    assert counts == {"CO_cd5": 2, "CO_W22_02": 2}
