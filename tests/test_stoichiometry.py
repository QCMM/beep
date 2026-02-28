"""Tests for beep/core/stoichiometry.py."""
from qcelemental.models.molecule import Molecule

from beep.core.stoichiometry import be_stoichiometry


def _make_struc_mol(ws3_cluster, h2_mol):
    """Build a combined structure from cluster + small molecule."""
    import numpy as np
    symbols = list(ws3_cluster.symbols) + list(h2_mol.symbols)
    # Concatenate geometries (in bohr)
    geom = np.concatenate([ws3_cluster.geometry, h2_mol.geometry]).flatten()
    return Molecule(symbols=symbols, geometry=geom)


def test_be_stoichiometry_keys(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert set(result.keys()) == {"default", "be_nocp", "ie", "de"}


def test_be_stoichiometry_default_count(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert len(result["default"]) == 7


def test_be_stoichiometry_be_nocp_count(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert len(result["be_nocp"]) == 3


def test_be_stoichiometry_ie_count(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert len(result["ie"]) == 3


def test_be_stoichiometry_de_count(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert len(result["de"]) == 4


def test_be_stoichiometry_coefficients(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    # default: +1, +1, +1, -1, -1, -1, -1
    coeffs = [c for _, c in result["default"]]
    assert coeffs == [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]


def test_be_stoichiometry_atom_counts(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    total_atoms = len(struc.symbols)
    # be_nocp: full structure, cluster, small molecule
    for mol, _ in result["be_nocp"]:
        assert len(mol.symbols) <= total_atoms
