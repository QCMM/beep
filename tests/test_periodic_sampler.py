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
    build_freeze_constraints,
    build_grid,
    find_cavity_z,
    frozen_atom_indices,
    hemisphere_z_shift,
    min_image_distance,
    min_image_vec,
    nearest_surface_atom,
    recenter_adsorbate_com,
    strip_adsorbate,
    wrap_into_cell,
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

def test_build_freeze_constraints_json_form():
    """geomeTRIC's JSON API takes the structured form, not the rendered text."""
    assert build_freeze_constraints([0, 1, 2]) == {
        "freeze": [{"type": "xyz", "indices": [0, 1, 2]}]
    }
    assert build_freeze_constraints([]) is None


def test_build_freeze_constraints_deduplicates_and_sorts():
    assert build_freeze_constraints([5, 5, 1, 3, 2]) == {
        "freeze": [{"type": "xyz", "indices": [1, 2, 3, 5]}]
    }


def test_build_freeze_constraints_indices_stay_zero_based():
    """geomeTRIC does the 0->1 shift itself; shifting here would freeze the wrong atoms."""
    out = build_freeze_constraints([0, 4])
    assert out["freeze"][0]["indices"] == [0, 4]


def test_build_freeze_constraints_accepted_by_geometric():
    """Regression guard for the crash that killed every frozen-slab periodic opt:
    geomeTRIC's run_json called .items() on a pre-rendered '$freeze ...' string and
    raised AttributeError before the first gradient. Feed our output to geomeTRIC's
    own renderer and require the classic block back."""
    run_json = pytest.importorskip("geometric.run_json")
    rendered = run_json.make_constraints_string(
        build_freeze_constraints([0, 1, 2, 5, 7, 8])
    )
    assert "$freeze" in rendered
    assert "xyz 1-3,6,8-9" in rendered   # 0-based in, 1-based rendered by geomeTRIC


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
    """Placement directly above a surface atom returns a tuple (mol, orig_ads_coords)."""
    cell_diag = np.array([10.0, 10.0, 30.0])
    pbc = [True, True, False]
    surface = _tiny_slab(cell_diag)
    adsorbate = _tiny_adsorbate()
    rng = random.Random(0)
    result = generate_candidate(
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
    assert result is not None
    mol, orig_ads = result
    assert list(mol.symbols) == ["O", "O", "O", "C", "O"]
    # last two atoms (adsorbate) sit above the slab in the centered molecule
    ads_geom = mol.geometry[3:].reshape(-1, 3)
    assert (ads_geom[:, 2] > 0).all()
    # original adsorbate coords should be the *pre-shift* placement (2 atoms)
    assert orig_ads.shape == (2, 3)


def test_generate_candidate_recenters_adsorbate_to_cell_center():
    """After generate_candidate, the adsorbate COM sits at (Lx/2, Ly/2)."""
    cell_diag = np.array([10.0, 10.0, 30.0])
    pbc = [True, True, False]
    surface = _tiny_slab(cell_diag)
    adsorbate = _tiny_adsorbate()
    rng = random.Random(0)
    result = generate_candidate(
        surface, adsorbate,
        x_bohr=2.0 * ANG2BOHR, y_bohr=2.0 * ANG2BOHR,   # away from center
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
    mol, _ = result
    n_surf = len(surface.symbols)
    ads_com = mol.geometry[n_surf:].reshape(-1, 3).mean(axis=0)
    # xy at cell center; z untouched by the shift so it's still above the slab
    assert ads_com[0] == pytest.approx(0.5 * cell_diag[0])
    assert ads_com[1] == pytest.approx(0.5 * cell_diag[1])


def test_generate_candidate_skips_impossible_sanity():
    """If sanity_min_distance is unsatisfiable, returns None instead of looping forever."""
    cell_diag = np.array([10.0, 10.0, 30.0])
    pbc = [True, True, False]
    surface = _tiny_slab(cell_diag)
    adsorbate = _tiny_adsorbate()
    rng = random.Random(0)
    result = generate_candidate(
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
    assert result is None


def test_strip_adsorbate_returns_only_surface_atoms():
    """Given a combined slab+adsorbate, strip_adsorbate returns only the first
    n_surface_atoms — atom order + positions preserved bit-for-bit."""
    combined = qcel.models.Molecule(
        symbols=["O", "O", "O", "C", "O"],  # 3 surface + 2 adsorbate (CO)
        geometry=np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 4.13],
        ]).flatten(),
        fix_com=False, fix_orientation=False,
    )
    bare = strip_adsorbate(combined, n_surface_atoms=3)
    assert list(bare.symbols) == ["O", "O", "O"]
    assert bare.geometry.shape == (3, 3)
    np.testing.assert_allclose(bare.geometry, combined.geometry.reshape(-1, 3)[:3])


def test_recenter_adsorbate_com_shifts_only_periodic_axes():
    """Non-periodic z is unchanged; xy shifts atoms uniformly."""
    cell_diag = np.array([10.0, 10.0, 30.0])
    pbc = [True, True, False]
    # 2 surface atoms at z=0, 1 adsorbate atom at (2, 2, 3)
    geom = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 0.0],
        [2.0, 2.0, 3.0],
    ])
    out = recenter_adsorbate_com(geom, n_surface_atoms=2, cell_diag_bohr=cell_diag, pbc=pbc)
    # adsorbate COM (only 1 atom) was (2, 2) → should end at (5, 5)
    assert out[2, 0] == pytest.approx(5.0)
    assert out[2, 1] == pytest.approx(5.0)
    # z unchanged
    assert out[2, 2] == pytest.approx(3.0)
    # surface atoms shifted by the same (+3, +3, 0) then wrapped
    assert out[0, 0] == pytest.approx(3.0)
    assert out[0, 1] == pytest.approx(3.0)
    assert out[0, 2] == pytest.approx(0.0)
    # second surface atom at (5,5) → (8,8)
    assert out[1, 0] == pytest.approx(8.0)
    assert out[1, 1] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Config validation: cart coordsys is incompatible with slab freezing
# ---------------------------------------------------------------------------

def _periodic_config_kwargs(**over):
    base = dict(
        workflow="sampling_periodic",
        molecule="CO",
        surface_collection="npasw500",
        sampling_level_of_theory={"mace_model": "/tmp/model.model"},
    )
    base.update(over)
    return base


def test_cart_coordsys_rejected_when_freezing():
    """geomeTRIC raises 'Do not use constraints with Cartesian coordinates', and cart
    is numerically unreliable on large slabs -- catch it at config load, not mid-run."""
    from beep.models.sampling_periodic import SamplingPeriodicConfig
    with pytest.raises(ValueError, match="cannot be combined with slab freezing"):
        SamplingPeriodicConfig(**_periodic_config_kwargs(
            sampling_opt_keywords={"coordsys": "cart"}, freeze_below_z_ang=4.0))


def test_cart_coordsys_allowed_without_freezing():
    from beep.models.sampling_periodic import SamplingPeriodicConfig
    cfg = SamplingPeriodicConfig(**_periodic_config_kwargs(
        sampling_opt_keywords={"coordsys": "cart"}))
    assert cfg.sampling_opt_keywords["coordsys"] == "cart"


def test_default_tric_with_freezing_is_accepted():
    from beep.models.sampling_periodic import SamplingPeriodicConfig
    cfg = SamplingPeriodicConfig(**_periodic_config_kwargs(freeze_below_z_ang=4.0))
    assert cfg.freeze_below_z_ang == 4.0


def test_overlay_xyz_tolerates_overlapping_copies(tmp_path):
    """Regression: the coverage overlay is slab + every accepted adsorbate copy, so
    near-coincident copies from adjacent grid nodes are normal. Routing it through
    qcelemental's Molecule raised 'Following atoms are too close' and aborted the whole
    sampling run over a cosmetic artifact."""
    from beep.core.periodic_sampler import write_overlay_xyz
    symbols = ["O", "H", "H", "C", "O", "C", "O"]
    geom = np.array([
        [0, 0, 0], [1.8, 0, 0], [-0.45, 1.76, 0],
        [0, 0, 6.0], [0, 0, 8.1],
        [0.07, 0, 6.0], [0.07, 0, 8.1],      # 0.07 bohr from the previous copy
    ], dtype=float).flatten()
    out = tmp_path / "overlay.xyz"
    write_overlay_xyz(out, symbols, geom)
    lines = out.read_text().splitlines()
    assert int(lines[0]) == len(symbols)
    assert len(lines) == len(symbols) + 2
    assert lines[2].split()[0] == "O"


def test_opt_program_defaults_to_geometric():
    """Existing cluster behaviour must not change silently."""
    from beep.models.sampling_periodic import SamplingPeriodicConfig
    cfg = SamplingPeriodicConfig(**_periodic_config_kwargs())
    assert cfg.sampling_opt_program == "geometric"


def test_opt_program_can_select_ase():
    from beep.models.sampling_periodic import SamplingPeriodicConfig
    cfg = SamplingPeriodicConfig(**_periodic_config_kwargs(sampling_opt_program="ase"))
    assert cfg.sampling_opt_program == "ase"


def test_cart_freeze_restriction_is_geometric_only():
    """geomeTRIC refuses constraints in Cartesian coordinates; ASE has no such limit,
    so the validator must not block a cart/freeze combination under 'ase'."""
    from beep.models.sampling_periodic import SamplingPeriodicConfig
    cfg = SamplingPeriodicConfig(**_periodic_config_kwargs(
        sampling_opt_program="ase", sampling_opt_keywords={"coordsys": "cart"},
        freeze_below_z_ang=4.0))
    assert cfg.freeze_below_z_ang == 4.0
    with pytest.raises(ValueError, match="cannot be combined with slab freezing"):
        SamplingPeriodicConfig(**_periodic_config_kwargs(
            sampling_opt_program="geometric", sampling_opt_keywords={"coordsys": "cart"},
            freeze_below_z_ang=4.0))
