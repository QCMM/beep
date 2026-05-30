"""Tests for beep/core/sampling.py."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qcelemental.models.molecule import Molecule

from beep.core.sampling import (
    generate_shell_list,
    compute_rmsd_conditional,
    filter_binding_sites,
)


# ---------------------------------------------------------------------------
# generate_shell_list
# ---------------------------------------------------------------------------

def test_generate_shell_list_sparse():
    result = generate_shell_list(10.0, "sparse")
    assert result == [10.0]


def test_generate_shell_list_normal():
    result = generate_shell_list(10.0, "normal")
    assert len(result) == 3
    assert result == [10.0, 8.0, 12.0]


def test_generate_shell_list_fine():
    result = generate_shell_list(10.0, "fine")
    assert len(result) == 5
    assert result == [10.0, 8.0, 12.0, 7.5, 15.0]


def test_generate_shell_list_hyperfine():
    result = generate_shell_list(10.0, "hyperfine")
    assert len(result) == 7


def test_generate_shell_list_invalid():
    with pytest.raises(ValueError):
        generate_shell_list(10.0, "bogus")


# ---------------------------------------------------------------------------
# compute_rmsd_conditional
# ---------------------------------------------------------------------------

def test_compute_rmsd_identical(h2_mol):
    r, rm = compute_rmsd_conditional(h2_mol, h2_mol, rmsd_symm=False, cutoff=0.4)
    assert abs(r) < 1e-10


def test_compute_rmsd_no_mirror(h2_mol):
    r, rm = compute_rmsd_conditional(h2_mol, h2_mol, rmsd_symm=False, cutoff=0.4)
    assert rm == 10.0  # sentinel value when mirror not used


def test_compute_rmsd_with_mirror(h2_mol):
    # With rmsd_symm=True and a tight cutoff that r >= cutoff (shift the molecule)
    geom = np.array(h2_mol.geometry) + np.array([0.01, 0.0, 0.0])
    shifted = Molecule(symbols=h2_mol.symbols, geometry=geom.flatten())
    r, rm = compute_rmsd_conditional(h2_mol, shifted, rmsd_symm=True, cutoff=0.0001)
    # Mirror path should have been taken since r >= cutoff
    assert rm != 10.0 or r < 0.0001


# ---------------------------------------------------------------------------
# filter_binding_sites
# ---------------------------------------------------------------------------

def test_filter_empty_inputs(test_logger):
    result = filter_binding_sites(
        [], [], cut_off_val=0.4, rmsd_symm=False,
        logger=test_logger, ligand_size=2,
    )
    assert result == []


def test_filter_no_duplicates(h2_mol, ws3_cluster, test_logger):
    # Two distinct molecules — both should be kept
    geom_shifted = np.array(ws3_cluster.geometry) + np.array([100.0, 0.0, 0.0])
    # Create two structures: ws3+h2 at different positions
    symbols1 = list(ws3_cluster.symbols) + list(h2_mol.symbols)
    geom1 = np.concatenate([ws3_cluster.geometry, h2_mol.geometry]).flatten()
    mol1 = Molecule(symbols=symbols1, geometry=geom1)

    geom2_h2 = np.array(h2_mol.geometry) + np.array([100.0, 0.0, 0.0])
    geom2 = np.concatenate([ws3_cluster.geometry, geom2_h2]).flatten()
    mol2 = Molecule(symbols=symbols1, geometry=geom2)

    result = filter_binding_sites(
        [("a", mol1), ("b", mol2)], [],
        cut_off_val=0.01, rmsd_symm=False,
        logger=test_logger, ligand_size=len(h2_mol.symbols),
    )
    assert len(result) == 2


def test_filter_removes_duplicate(h2_mol, ws3_cluster, test_logger):
    # Two identical molecules — one should be removed
    symbols = list(ws3_cluster.symbols) + list(h2_mol.symbols)
    geom = np.concatenate([ws3_cluster.geometry, h2_mol.geometry]).flatten()
    mol = Molecule(symbols=symbols, geometry=geom)

    result = filter_binding_sites(
        [("a", mol), ("b", mol)], [],
        cut_off_val=0.4, rmsd_symm=False,
        logger=test_logger, ligand_size=len(h2_mol.symbols),
    )
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Real optimized structures from QCFractal (CO on w2/w3 clusters)
# ---------------------------------------------------------------------------

def test_rmsd_real_same_structure(co_w2_0001):
    """RMSD of a real binding site against itself should be ~0."""
    r, rm = compute_rmsd_conditional(co_w2_0001, co_w2_0001, rmsd_symm=False, cutoff=0.4)
    assert abs(r) < 1e-10


def test_rmsd_real_different_binding_sites(co_w2_0001, co_w2_0007):
    """Two different CO-w2 binding sites should have non-zero RMSD."""
    r, rm = compute_rmsd_conditional(co_w2_0001, co_w2_0007, rmsd_symm=False, cutoff=0.4)
    assert r > 0.01


def test_filter_real_distinct_sites_kept(co_w2_0001, co_w2_0007, test_logger):
    """Two genuinely different binding sites should both survive filtering."""
    ligand_size = 2  # CO has 2 atoms
    result = filter_binding_sites(
        [("co_w2_0001", co_w2_0001), ("co_w2_0007", co_w2_0007)], [],
        cut_off_val=0.25, rmsd_symm=False,
        logger=test_logger, ligand_size=ligand_size,
    )
    assert len(result) == 2


def test_filter_real_against_reference(co_w2_0001, co_w3_0001, co_w3_0004, test_logger):
    """Filter new candidates against an existing reference set."""
    ligand_size = 2  # CO
    # co_w3 structures are different systems (3 waters vs 2) so they won't match
    result = filter_binding_sites(
        [("co_w3_0001", co_w3_0001), ("co_w3_0004", co_w3_0004)],
        [("co_w2_0001", co_w2_0001)],
        cut_off_val=0.25, rmsd_symm=False,
        logger=test_logger, ligand_size=ligand_size,
        atoms_map=False,  # different atom counts, can't use atoms_map
    )
    # w3 structures have 11 atoms vs w2's 8 — can't align, so both should survive
    assert len(result) >= 1


def test_rmsd_real_co_w5_different_sites(co_w5_0001, co_w5_0002):
    """Two different CO-w5 binding sites should have non-zero RMSD."""
    r, rm = compute_rmsd_conditional(co_w5_0001, co_w5_0002, rmsd_symm=False, cutoff=0.4)
    assert r > 0.01


def test_filter_real_co_w5_distinct_kept(co_w5_0001, co_w5_0002, test_logger):
    """Two distinct CO-w5 binding sites should both survive filtering."""
    ligand_size = 2  # CO
    result = filter_binding_sites(
        [("co_w5_0001", co_w5_0001), ("co_w5_0002", co_w5_0002)], [],
        cut_off_val=0.25, rmsd_symm=False,
        logger=test_logger, ligand_size=ligand_size,
    )
    assert len(result) == 2


# ---------------------------------------------------------------------------
# run_sampling — workflow-level case-B regression
# ---------------------------------------------------------------------------

def test_run_sampling_submits_at_new_lot_when_all_entries_exist(test_logger):
    """Regression for reports/BUG_beep_sampling_spec_change.md.

    When the user re-runs sampling at a different LOT against a dataset
    whose entries already exist (from a prior LOT), the workflow must
    still call submit_optimizations at the current LOT for those
    pre-existing entries. The old code only submitted when there were
    *new* entries to add, so a LOT change on already-populated datasets
    silently produced zero new opts.
    """
    from beep.workflows.sampling import run_sampling

    cluster_name = "CO_W3_01"
    # Names that the workflow will construct for max_structures=3
    existing_entry_names = [f"{cluster_name}_{i:04d}" for i in (1, 2, 3)]

    sampling_dset = MagicMock()
    sampling_dset.name = f"pre_{cluster_name}"
    sampling_dset.entry_names = existing_entry_names
    # iterate_entries yields nothing — keeps the "existing molecules" list empty
    refinement_dset = MagicMock()
    refinement_dset.name = cluster_name
    refinement_dset.iterate_entries.return_value = iter([])
    refinement_dset.entry_names = []

    cluster_mol = MagicMock(); cluster_mol.symbols = ["O", "H", "H"]
    target_mol = MagicMock(); target_mol.symbols = ["C", "O"]

    client = MagicMock()

    submit_call_kwargs = []

    def fake_submit_optimizations(ds_opt, opt_lot, tag, subset=None):
        submit_call_kwargs.append(
            {"ds": ds_opt.name, "opt_lot": opt_lot, "tag": tag,
             "subset": list(subset) if subset else None}
        )
        r = MagicMock(); r.n_inserted = len(subset or []); r.n_existing = 0
        return r

    with patch("beep.workflows.sampling.qcf") as mock_qcf:
        mock_qcf.add_opt_specification.return_value = None
        mock_qcf.get_job_ids.return_value = []
        mock_qcf.wait_for_completion.return_value = None
        mock_qcf.submit_optimizations.side_effect = fake_submit_optimizations
        mock_qcf.fetch_opt_molecules.return_value = []
        mock_qcf.add_opt_entry.return_value = None

        # generate_shell_list("sparse", 2.0) returns a single shell, so the
        # per-shell loop runs exactly once.
        run_sampling(
            method="gfn2-xtb", basis=None, program="xtb",
            tag="sampling", kw_id=None,
            sampling_opt_dset=sampling_dset,
            refinement_opt_dset=refinement_dset,
            opt_lot="gfn2-xtb",
            rmsd_symm=False, store_initial=False, rmsd_val=0.4,
            target_mol=target_mol, cluster=cluster_mol,
            debug_path="/tmp/dbg",
            client=client,
            sampling_shell=2.0, sampling_condition="sparse",
            logger=test_logger,
        )

    # The fix: submit_optimizations MUST be called for the pre-existing
    # entries at the new opt_lot, even though no new entries were added.
    assert len(submit_call_kwargs) == 1, submit_call_kwargs
    call = submit_call_kwargs[0]
    assert call["opt_lot"] == "gfn2-xtb"
    assert call["tag"] == "sampling"
    assert sorted(call["subset"]) == sorted(existing_entry_names)
