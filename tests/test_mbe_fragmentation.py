"""Tests for beep/core/mbe_fragmentation.py.

Ported from beep-mbe's test_fragmentation.py; the import path and exception
type change, and one case exercises a real water-cluster fixture.
"""
import pytest
from qcelemental.models import Molecule

from beep.core.exceptions import MbeFragmentationError
from beep.core.mbe_fragmentation import (
    fragment_cluster,
    fragment_small_molecule,
    fragment_surface_model,
)


def _make_molecule(natoms: int) -> Molecule:
    coords = "\n".join([f"H {i}.0 0.0 0.0" for i in range(natoms)])
    xyz = f"{natoms}\n\n{coords}\n"
    return Molecule.from_data(xyz)


def test_surface_divisibility_error():
    mol = _make_molecule(5)
    with pytest.raises(MbeFragmentationError):
        fragment_surface_model(mol, env_unit_len=3)


def test_cluster_last_fragment_indices():
    mol = _make_molecule(9)
    fragmented = fragment_cluster(mol, env_unit_len=3, small_molecule_atoms=3)
    assert [list(fragment) for fragment in fragmented.fragments] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]


def test_small_molecule_single_fragment():
    mol = _make_molecule(4)
    fragmented = fragment_small_molecule(mol)
    assert [list(f) for f in fragmented.fragments] == [[0, 1, 2, 3]]
    assert list(fragmented.fragment_charges) == [0]
    assert list(fragmented.fragment_multiplicities) == [1]


def test_cluster_smaller_than_adsorbate_raises():
    mol = _make_molecule(2)
    with pytest.raises(MbeFragmentationError):
        fragment_cluster(mol, env_unit_len=3, small_molecule_atoms=3)


def test_surface_model_real_water_cluster(ws3_cluster):
    """A 3-water surface model (9 atoms) fragments into three 3-atom units."""
    fragmented = fragment_surface_model(ws3_cluster, env_unit_len=3)
    assert len(fragmented.fragments) == 3
    assert all(len(f) == 3 for f in fragmented.fragments)


# ---------------------------------------------------------------------------
# Charge / multiplicity propagation (radical adsorbates must not become singlets)
# ---------------------------------------------------------------------------

# Water dimer (6 atoms) followed by HCO (3 atoms, doublet). Charge/multiplicity
# line "0 2" sets molecular_multiplicity=2 on the whole cluster.
_HCO_ON_W2_XYZ = """0 2
O   -1.551007  -0.114520   0.000000
H   -1.934259   0.762503   0.000000
H   -0.599677   0.040712   0.000000
O    1.350625   0.111469   0.000000
H    1.680398  -0.373741  -0.758561
H    1.680398  -0.373741   0.758561
C    0.000000   3.500000   0.000000
O    1.170000   3.500000   0.000000
H   -0.600000   4.450000   0.000000
"""


def test_cluster_radical_adsorbate_keeps_doublet_multiplicity():
    """Regression: fragment_cluster hard-coded multiplicity 1 for every
    fragment, so an HCO adsorbate on a water cluster was made a singlet."""
    mol = Molecule.from_data(_HCO_ON_W2_XYZ, dtype="psi4")
    assert mol.molecular_multiplicity == 2
    fragmented = fragment_cluster(mol, env_unit_len=3, small_molecule_atoms=3)
    assert [list(f) for f in fragmented.fragments] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert list(fragmented.fragment_multiplicities) == [1, 1, 2]
    assert list(fragmented.fragment_charges) == [0, 0, 0]
    assert fragmented.molecular_multiplicity == 2


def test_small_molecule_radical_keeps_doublet(hco_mol):
    """fragment_small_molecule must carry the molecule's own multiplicity."""
    data = hco_mol.dict()
    data["molecular_multiplicity"] = 2
    data["fragment_multiplicities"] = [2]
    hco = Molecule(**data)
    fragmented = fragment_small_molecule(hco)
    assert list(fragmented.fragment_multiplicities) == [2]
    assert list(fragmented.fragment_charges) == [0]


def test_cluster_uses_parent_fragment_state_when_partition_matches():
    """If the input already carries per-fragment charge/multiplicity on the
    same partition, those values are kept verbatim."""
    mol = Molecule.from_data(_HCO_ON_W2_XYZ, dtype="psi4")
    data = mol.dict()
    data.update({
        "fragments": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "fragment_charges": [0, 0, 0],
        "fragment_multiplicities": [1, 1, 2],
    })
    pre = Molecule(**data)
    fragmented = fragment_cluster(pre, env_unit_len=3, small_molecule_atoms=3)
    assert list(fragmented.fragment_multiplicities) == [1, 1, 2]


def test_closed_shell_cluster_still_all_singlets(ws3_cluster):
    """Closed-shell neutral inputs keep the historical 0/1 assignment."""
    fragmented = fragment_cluster(ws3_cluster, env_unit_len=3, small_molecule_atoms=3)
    assert list(fragmented.fragment_multiplicities) == [1, 1, 1]
    assert list(fragmented.fragment_charges) == [0, 0, 0]
    surf = fragment_surface_model(ws3_cluster, env_unit_len=3)
    assert list(surf.fragment_multiplicities) == [1, 1, 1]
