"""Unit tests for the periodic_sampler helpers.

Pure-python (no QCFractal / MACE required). Uses qcelemental only for the
Molecule construction that a couple of helpers need.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import qcelemental as qcel

from beep.core.periodic_sampler import (
    ANG2BOHR,
    BOHR2ANG,
    all_atoms_ok,
    build_freeze_constraint_string,
    build_grid,
    find_cavity_z,
    frozen_atom_indices,
    hemisphere_z_shift,
    min_image_distance,
    min_image_vec,
    nearest_surface_atom,
    wrap_into_cell,
    _atom_index_ranges,
    _cell_diag_bohr,
    generate_candidate,
)


# ---------------------------------------------------------------------------
# PBC utilities
# ---------------------------------------------------------------------------

def test_min_image_vec_wraps_across_edge():
    cell = np.array([10.0, 10.0, 30.0])   # bohr
    pbc = [True, True, False]
    # displacement of 9 in x should be wrapped to -1
    dp = np.array([9.0, 0.0, 0.0])
    assert np.allclose(min_image_vec(dp, cell, pbc), [-1.0, 0.0, 0.0])
    # displacement of 15 in z (non-periodic) should NOT wrap
    dp = np.array([0.0, 0.0, 15.0])
    assert np.allclose(min_image_vec(dp, cell, pbc), [0.0, 0.0, 15.0])


def test_min_image_distance_across_boundary():
    cell = np.array([10.0, 10.0, 30.0])
    pbc = [True, True, False]
    p1 = np.array([0.5, 5.0, 5.0])
    p2 = np.array([9.5, 5.0, 5.0])
    # naive Euclidean would say 9.0, min-image should be 1.0
    assert min_image_distance(p1, p2, cell, pbc) == pytest.approx(1.0)


def test_wrap_into_cell_pbc_only():
    cell = np.array([10.0, 10.0, 30.0])
    pbc = [True, True, False]
    coords = np.array([[11.0, -1.0, 40.0], [0.5, 0.5, -5.0]])
    wrapped = wrap_into_cell(coords, cell, pbc)
    # x/y wrap; z untouched
    assert np.allclose(wrapped[0], [1.0, 9.0, 40.0])
    assert np.allclose(wrapped[1], [0.5, 0.5, -5.0])


def test_cell_diag_conversion():
    cell_ang = [[10.0, 0, 0], [0, 20.0, 0], [0, 0, 30.0]]
    diag = _cell_diag_bohr(cell_ang)
    assert np.allclose(diag, np.array([10.0, 20.0, 30.0]) * ANG2BOHR)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def test_build_grid_covers_full_footprint():
    """Grid starts at 0 (full-footprint) — differs from the monoliths' 1 A border."""
    cell_diag = np.array([15.0, 15.0, 30.0])
    rng = random.Random(0)
    x, y = build_grid(cell_diag, step_size_bohr=3.0, noise_frac=0.0, rng=rng)
    assert x[0] == 0.0
    assert x[-1] < 15.0            # never exceeds Lx (arange half-open)
    # 5 nodes: 0, 3, 6, 9, 12
    assert len(x) == 5
    assert np.allclose(x, [0.0, 3.0, 6.0, 9.0, 12.0])


def test_build_grid_leaves_boundaries_unperturbed():
    cell_diag = np.array([15.0, 15.0, 30.0])
    rng = random.Random(0)
    x, y = build_grid(cell_diag, step_size_bohr=3.0, noise_frac=0.25, rng=rng)
    # boundary nodes (first and last) must be unchanged
    assert x[0] == 0.0
    assert x[-1] == pytest.approx(12.0)  # last element from arange(0,15,3)
    # interior nodes should have moved by at most ±0.25*3 = 0.75 bohr
    for i in range(1, len(x) - 1):
        assert abs(x[i] - (i * 3.0)) <= 0.25 * 3.0


def test_build_grid_reproducible_with_seed():
    cell_diag = np.array([15.0, 15.0, 30.0])
    x1, _ = build_grid(cell_diag, 3.0, 0.25, random.Random(42))
    x2, _ = build_grid(cell_diag, 3.0, 0.25, random.Random(42))
    assert np.allclose(x1, x2)


# ---------------------------------------------------------------------------
# Placement primitives
# ---------------------------------------------------------------------------

def test_hemisphere_z_shift_positive_and_none():
    d = 2.5
    assert hemisphere_z_shift(d, 0.0) == pytest.approx(2.5)
    assert hemisphere_z_shift(d, 1.5) == pytest.approx(np.sqrt(2.5**2 - 1.5**2))
    assert hemisphere_z_shift(d, 3.0) is None   # xy_dist > d


def test_nearest_surface_atom_with_pbc():
    """A point near a cell edge should find the periodic image of the far-side atom."""
    cell_diag = np.array([10.0, 10.0, 30.0])
    pbc = [True, True, False]
    # Surface atom near x=9.5 (mirror image at -0.5)
    surface_geom = np.array([[9.5, 5.0, 5.0], [5.0, 5.0, 5.0]])
    # Probe at x=0.5 — non-PBC nearest would be the second atom (at 5.0),
    # but under PBC the first atom's image at -0.5 is closer.
    idx, d = nearest_surface_atom(surface_geom, np.array([0.5, 5.0, 5.0]), cell_diag, pbc)
    assert idx == 0
    assert d == pytest.approx(1.0)


def test_find_cavity_z_picks_best_fit():
    """Best-fit z (not first hit) inside a widened window."""
    # One surface atom at z=0; scan from z_top down; at z=2.5 the distance
    # is exactly the target 2.5; at z=3.0 the distance is 3.0. With window=1.0
    # the acceptance band is [1.5, 2.5], so only z=2.5 qualifies (best-fit).
    cell_diag = np.array([50.0, 50.0, 50.0])
    pbc = [True, True, False]
    surface_geom = np.array([[5.0, 5.0, 0.0]])
    best_z = find_cavity_z(
        surface_geom, x=5.0, y=5.0, z_range_bohr=(0.0, 5.0),
        scan_step_bohr=0.5, sampling_distance_bohr=2.5, window_bohr=1.0,
        cell_diag_bohr=cell_diag, pbc=pbc,
    )
    assert best_z == pytest.approx(2.5)


def test_find_cavity_z_returns_none_when_no_z_qualifies():
    # Nearest atom always more than 5 bohr away → no z in [1.5, 2.5]
    cell_diag = np.array([50.0, 50.0, 50.0])
    pbc = [True, True, False]
    surface_geom = np.array([[30.0, 30.0, 0.0]])
    best_z = find_cavity_z(
        surface_geom, x=5.0, y=5.0, z_range_bohr=(0.0, 3.0),
        scan_step_bohr=0.5, sampling_distance_bohr=2.5, window_bohr=1.0,
        cell_diag_bohr=cell_diag, pbc=pbc,
    )
    assert best_z is None


def test_all_atoms_ok_flags_overlap():
    cell_diag = np.array([30.0, 30.0, 30.0])
    pbc = [True, True, False]
    surface_geom = np.array([[10.0, 10.0, 0.0]])
    # An adsorbate atom right on the surface atom → not ok
    ads_close = np.array([[10.0, 10.0, 1.0]])   # 1 bohr away < 1.5 A (~2.83 bohr)
    assert not all_atoms_ok(ads_close, surface_geom, cell_diag, pbc, min_dist_bohr=1.5 * ANG2BOHR)
    # Well away → ok
    ads_far = np.array([[10.0, 10.0, 10.0]])
    assert all_atoms_ok(ads_far, surface_geom, cell_diag, pbc, min_dist_bohr=1.5 * ANG2BOHR)


# ---------------------------------------------------------------------------
# Freeze constraints for geomeTRIC
# ---------------------------------------------------------------------------

def test_atom_index_ranges_compresses_runs():
    assert _atom_index_ranges([0, 1, 2, 3, 5, 7, 8, 9]) == "1-4,6,8-10"


def test_atom_index_ranges_deduplicates_and_sorts():
    # input is 0-indexed; helper emits 1-indexed for geomeTRIC's constraints format
    # {5,5,1,3,2} → dedup+shift → {2,3,4,6} → "2-4,6"
    assert _atom_index_ranges([5, 5, 1, 3, 2]) == "2-4,6"


def test_build_freeze_constraint_string_format():
    s = build_freeze_constraint_string([0, 1, 2])
    assert s == "$freeze\nxyz 1-3\n"
    assert build_freeze_constraint_string([]) is None


def test_frozen_atom_indices_from_z_threshold():
    """freeze_below_z_ang: freeze slab atoms whose z is below the threshold."""
    # surface_geom is in bohr; threshold is in Angstrom
    surface_geom = np.array([
        [0.0, 0.0, 0.0],           # z=0 A
        [1.0, 0.0, 2.0 * ANG2BOHR],  # z=2 A
        [0.0, 1.0, 5.0 * ANG2BOHR],  # z=5 A
    ])
    got = frozen_atom_indices(surface_geom, freeze_below_z_ang=3.0, freeze_atoms=None, n_surface_atoms=3)
    assert got == [0, 1]


def test_frozen_atom_indices_explicit_overrides_threshold():
    surface_geom = np.array([[0.0, 0.0, 0.0]])
    got = frozen_atom_indices(
        surface_geom, freeze_below_z_ang=3.0, freeze_atoms=[7, 8], n_surface_atoms=1,
    )
    assert got == [7, 8]


def test_frozen_atom_indices_no_freeze_when_both_none():
    surface_geom = np.array([[0.0, 0.0, 0.0]])
    got = frozen_atom_indices(surface_geom, freeze_below_z_ang=None, freeze_atoms=None, n_surface_atoms=1)
    assert got == []


# ---------------------------------------------------------------------------
# End-to-end candidate generation on a tiny synthetic slab
# ---------------------------------------------------------------------------

def _tiny_slab(cell_diag_bohr) -> qcel.models.Molecule:
    """3-oxygen slab occupying the xy plane at z=0."""
    return qcel.models.Molecule(
        symbols=["O", "O", "O"],
        geometry=np.array([
            [2.0 * ANG2BOHR, 2.0 * ANG2BOHR, 0.0],
            [5.0 * ANG2BOHR, 2.0 * ANG2BOHR, 0.0],
            [2.0 * ANG2BOHR, 5.0 * ANG2BOHR, 0.0],
        ]).flatten(),
        fix_com=False,
        fix_orientation=False,
    )


def _tiny_adsorbate() -> qcel.models.Molecule:
    """Diatomic CO for testing."""
    return qcel.models.Molecule(
        symbols=["C", "O"],
        geometry=np.array([[0.0, 0.0, 0.0], [1.13 * ANG2BOHR, 0.0, 0.0]]).flatten(),
        fix_com=False,
        fix_orientation=False,
    )


def test_generate_candidate_happy_path():
    """Placement directly above a surface atom returns a molecule (surface + adsorbate)."""
    cell_diag = np.array([10.0, 10.0, 30.0])
    pbc = [True, True, False]
    surface = _tiny_slab(cell_diag)
    adsorbate = _tiny_adsorbate()
    rng = random.Random(0)
    mol = generate_candidate(
        surface, adsorbate,
        x_bohr=2.0 * ANG2BOHR, y_bohr=2.0 * ANG2BOHR,
        z_top_bohr=5.0 * ANG2BOHR,
        z_scan_range_bohr=(0.0, 5.0 * ANG2BOHR),
        sampling_distance_bohr=2.5 * ANG2BOHR,
        cell_diag_bohr=cell_diag, pbc=pbc,
        cavity_scan_step_bohr=0.5 * ANG2BOHR,
        cavity_window_bohr=1.0 * ANG2BOHR,
        sanity_min_dist_bohr=1.5 * ANG2BOHR,
        sanity_max_iter=20,
        rng=rng,
    )
    assert mol is not None
    assert list(mol.symbols) == ["O", "O", "O", "C", "O"]
    # last two atoms (adsorbate) sit above the slab
    ads_geom = mol.geometry[3:].reshape(-1, 3)
    assert (ads_geom[:, 2] > 0).all()


def test_generate_candidate_skips_impossible_sanity():
    """If sanity_min_distance is unsatisfiable, returns None instead of looping forever."""
    cell_diag = np.array([10.0, 10.0, 30.0])
    pbc = [True, True, False]
    surface = _tiny_slab(cell_diag)
    adsorbate = _tiny_adsorbate()
    rng = random.Random(0)
    mol = generate_candidate(
        surface, adsorbate,
        x_bohr=2.0 * ANG2BOHR, y_bohr=2.0 * ANG2BOHR,
        z_top_bohr=5.0 * ANG2BOHR,
        z_scan_range_bohr=(0.0, 5.0 * ANG2BOHR),
        sampling_distance_bohr=0.1 * ANG2BOHR,   # unphysically close
        cell_diag_bohr=cell_diag, pbc=pbc,
        cavity_scan_step_bohr=0.5 * ANG2BOHR,
        cavity_window_bohr=1.0 * ANG2BOHR,
        sanity_min_dist_bohr=5.0 * ANG2BOHR,     # impossibly large
        sanity_max_iter=5,
        rng=rng,
    )
    assert mol is None
