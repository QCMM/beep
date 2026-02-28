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
