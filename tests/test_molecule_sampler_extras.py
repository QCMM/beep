"""Tests for previously untested functions in beep/core/molecule_sampler.py.

Covers: com, surface_distance_check, create_debug_molecule,
        single_site_spherical_sampling.
"""
import numpy as np
import pytest
from qcelemental.models.molecule import Molecule

from beep.core.molecule_sampler import (
    com,
    surface_distance_check,
    create_debug_molecule,
    single_site_spherical_sampling,
    angst2bohr,
)


# ---------------------------------------------------------------------------
# com (center of mass)
# ---------------------------------------------------------------------------

def test_com_single_atom():
    """COM of a single atom is its own position."""
    geom = np.array([[1.0, 2.0, 3.0]])
    result = com(geom, ["O"])
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_com_symmetric_homonuclear():
    """COM of two identical atoms is their midpoint."""
    geom = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = com(geom, ["H", "H"])
    np.testing.assert_allclose(result, [1.0, 0.0, 0.0])


def test_com_weighted():
    """COM is mass-weighted — heavier atom pulls the center toward it."""
    # O at origin, H at (10, 0, 0)
    geom = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    result = com(geom, ["O", "H"])
    # O mass ~ 15.999, H mass ~ 1.008 → COM should be close to O
    assert result[0] < 1.0  # much closer to O than to H
    assert result[0] > 0.0  # but not at O exactly


def test_com_h2_fixture(h2_mol):
    """COM of H2 molecule from fixture is the midpoint of the two H atoms."""
    result = com(h2_mol.geometry, list(h2_mol.symbols))
    midpoint = np.mean(h2_mol.geometry, axis=0)
    np.testing.assert_allclose(result, midpoint, atol=1e-10)


def test_com_ws3_shape(ws3_cluster):
    """COM returns a (3,) array for a multi-atom cluster."""
    result = com(ws3_cluster.geometry, list(ws3_cluster.symbols))
    assert result.shape == (3,)


# ---------------------------------------------------------------------------
# surface_distance_check
# ---------------------------------------------------------------------------

def test_surface_distance_check_far_apart():
    """Molecules far apart should pass the check (return True)."""
    cluster = Molecule(symbols=["O", "H", "H"],
                       geometry=[0, 0, 0, 1, 0, 0, 0, 1, 0])
    mol = Molecule(symbols=["H", "H"],
                   geometry=[100, 100, 100, 101, 100, 100])
    assert surface_distance_check(cluster, mol, cut_distance=2.0) is True


def test_surface_distance_check_too_close():
    """Atoms within the cutoff should fail the check (return False)."""
    cluster = Molecule(symbols=["O", "H", "H"],
                       geometry=[0, 0, 0, 1, 0, 0, 0, 1, 0])
    # Place mol close to cluster but far enough for QCElemental validation
    # 2 bohr apart ~ 1.06 Angstrom — within a 2.0 Ang cutoff
    mol = Molecule(symbols=["H", "H"],
                   geometry=[2.0, 0, 0, 3.5, 0, 0])
    assert surface_distance_check(cluster, mol, cut_distance=2.0) is False


def test_surface_distance_check_boundary(h2_mol, ws3_cluster):
    """With a very large cutoff, everything should fail."""
    assert surface_distance_check(ws3_cluster, h2_mol, cut_distance=1000.0) is False


def test_surface_distance_check_zero_cutoff(h2_mol, ws3_cluster):
    """With cutoff=0, nothing can be too close (unless exactly overlapping)."""
    assert surface_distance_check(ws3_cluster, h2_mol, cut_distance=0.0) is True


# ---------------------------------------------------------------------------
# create_debug_molecule
# ---------------------------------------------------------------------------

def test_create_debug_molecule_atom_count(ws3_cluster, h2_mol):
    """Debug molecule should have cluster + all sampled molecule atoms."""
    shifted1 = h2_mol.scramble(
        do_shift=np.array([10.0, 0.0, 0.0]), do_rotate=False, do_resort=False
    )[0]
    shifted2 = h2_mol.scramble(
        do_shift=np.array([0.0, 10.0, 0.0]), do_rotate=False, do_resort=False
    )[0]
    debug_mol = create_debug_molecule(ws3_cluster, [shifted1, shifted2])
    expected_atoms = len(ws3_cluster.symbols) + 2 * len(h2_mol.symbols)
    assert len(debug_mol.symbols) == expected_atoms


def test_create_debug_molecule_empty_list(ws3_cluster):
    """Debug molecule with no sampled molecules is just the cluster."""
    debug_mol = create_debug_molecule(ws3_cluster, [])
    assert len(debug_mol.symbols) == len(ws3_cluster.symbols)
    np.testing.assert_allclose(
        debug_mol.geometry[:len(ws3_cluster.symbols)],
        ws3_cluster.geometry,
        atol=1e-8,
    )


def test_create_debug_molecule_preserves_cluster_geom(ws3_cluster, h2_mol):
    """The cluster geometry should appear at the beginning of the debug molecule."""
    shifted = h2_mol.scramble(
        do_shift=np.array([20.0, 0.0, 0.0]), do_rotate=False, do_resort=False
    )[0]
    debug_mol = create_debug_molecule(ws3_cluster, [shifted])
    n_cluster = len(ws3_cluster.symbols)
    np.testing.assert_allclose(
        debug_mol.geometry[:n_cluster],
        ws3_cluster.geometry,
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# single_site_spherical_sampling
# ---------------------------------------------------------------------------

def test_single_site_spherical_sampling_returns_molecules(co_w2_0001):
    """Should return a list of Molecule objects."""
    # co_w2_0001 has 8 atoms: 6 water + 2 CO
    # Use H2 as sampling molecule (2 atoms)
    sampling_mol = Molecule(
        symbols=list(co_w2_0001.symbols[6:]),
        geometry=co_w2_0001.geometry[6:].flatten(),
    )
    result = single_site_spherical_sampling(
        cluster=co_w2_0001,
        sampling_mol=sampling_mol,
        sampled_mol_size=2,
        sampling_shell=3.0,
        grid_size="sparse",
        purge=False,
        noise=False,
        zenith_angle=np.pi,
        print_out=False,
    )
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(m, Molecule) for m in result)


def test_single_site_spherical_sampling_atom_counts(co_w2_0001):
    """Each output molecule should have cluster_atoms + sampling_mol_atoms."""
    sampling_mol = Molecule(
        symbols=list(co_w2_0001.symbols[6:]),
        geometry=co_w2_0001.geometry[6:].flatten(),
    )
    result = single_site_spherical_sampling(
        cluster=co_w2_0001,
        sampling_mol=sampling_mol,
        sampled_mol_size=2,
        sampling_shell=3.0,
        grid_size="sparse",
        purge=False,
        noise=False,
        zenith_angle=np.pi,
        print_out=False,
    )
    expected = len(co_w2_0001.symbols) + len(sampling_mol.symbols)
    for mol in result:
        assert len(mol.symbols) == expected


def test_single_site_spherical_sampling_grid_sizes(co_w2_0001):
    """Denser grids should produce more structures."""
    sampling_mol = Molecule(
        symbols=list(co_w2_0001.symbols[6:]),
        geometry=co_w2_0001.geometry[6:].flatten(),
    )
    counts = {}
    for grid in ["sparse", "normal", "dense"]:
        result = single_site_spherical_sampling(
            cluster=co_w2_0001,
            sampling_mol=sampling_mol,
            sampled_mol_size=2,
            sampling_shell=3.0,
            grid_size=grid,
            purge=False,
            noise=False,
            zenith_angle=np.pi,
            print_out=False,
        )
        counts[grid] = len(result)
    assert counts["sparse"] < counts["normal"] < counts["dense"]


def test_single_site_spherical_sampling_with_noise(co_w2_0001):
    """Noise=True should still produce valid molecules."""
    sampling_mol = Molecule(
        symbols=list(co_w2_0001.symbols[6:]),
        geometry=co_w2_0001.geometry[6:].flatten(),
    )
    result = single_site_spherical_sampling(
        cluster=co_w2_0001,
        sampling_mol=sampling_mol,
        sampled_mol_size=2,
        sampling_shell=3.0,
        grid_size="sparse",
        purge=False,
        noise=True,
        zenith_angle=np.pi,
        print_out=False,
    )
    assert len(result) > 0
    assert all(isinstance(m, Molecule) for m in result)
