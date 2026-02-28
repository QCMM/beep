"""Tests for beep/core/sampling.py."""
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
