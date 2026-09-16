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
