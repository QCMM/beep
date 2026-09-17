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


# ---------------------------------------------------------------------------
# Periodic duplicate-site filter
# ---------------------------------------------------------------------------

def _ads(symbols, positions_ang, n_slab=0):
    """Molecule with n_slab dummy slab atoms first, adsorbate last (BEEP convention)."""
    import qcelemental as qcel
    B = 1.8897259886
    pos = [[10.0, 10.0, 0.0]] * n_slab + list(positions_ang)
    return qcel.models.Molecule(
        symbols=["He"] * n_slab + list(symbols),
        geometry=(np.array(pos) * B).flatten(),
        fix_com=False, fix_orientation=False, validate=False,
    )


CELL = [[31.0, 0, 0], [0, 31.0, 0], [0, 0, 60.0]]
PBC = [True, True, False]


def test_periodic_filter_catches_wraparound_duplicates():
    """The cluster filter has no cell: an adsorbate at x=0.3 and one at x=30.8 in a
    31 A cell are 0.5 A apart, but it measures 30.5 A and keeps both."""
    from beep.core.periodic_sampler import filter_periodic_sites

    a = ("a", _ads(["C", "O"], [[0.3, 5.0, 12.0], [0.3, 5.0, 13.13]]))
    b = ("b", _ads(["C", "O"], [[30.8, 5.0, 12.0], [30.8, 5.0, 13.13]]))
    unique = filter_periodic_sites([a, b], CELL, PBC, 2, com_tol_ang=1.0)
    assert len(unique) == 1, "wrap-around duplicate should collapse to one site"


def test_periodic_filter_keeps_distinct_positions():
    from beep.core.periodic_sampler import filter_periodic_sites

    a = ("a", _ads(["C", "O"], [[5.0, 5.0, 12.0], [5.0, 5.0, 13.13]]))
    b = ("b", _ads(["C", "O"], [[15.0, 5.0, 12.0], [15.0, 5.0, 13.13]]))
    assert len(filter_periodic_sites([a, b], CELL, PBC, 2, com_tol_ang=1.0)) == 2


def test_periodic_filter_separates_binding_modes_at_one_position():
    """C-down and O-down CO at the same spot are different modes, not duplicates."""
    from beep.core.periodic_sampler import filter_periodic_sites

    c_down = ("c", _ads(["C", "O"], [[5.0, 5.0, 12.0], [5.0, 5.0, 13.13]]))
    o_down = ("o", _ads(["C", "O"], [[5.0, 5.0, 13.13], [5.0, 5.0, 12.0]]))
    assert len(filter_periodic_sites([c_down, o_down], CELL, PBC, 2,
                                     com_tol_ang=1.0, orient_tol_ang=0.3)) == 2
    # and with the orientation test disabled they merge
    assert len(filter_periodic_sites([c_down, o_down], CELL, PBC, 2,
                                     com_tol_ang=1.0, orient_tol_ang=None)) == 1


def test_periodic_filter_is_permutation_invariant():
    """Relabelling identical atoms must not create a new site (the failure that makes
    a 'longest interatomic vector' axis unusable for CH4/CH3)."""
    from beep.core.periodic_sampler import filter_periodic_sites

    t = 1.09 / np.sqrt(3)
    hs = [[t * 1.6, 0, t], [-t * 0.8, t * 1.4, t], [-t * 0.8, -t * 1.4, t], [0, 0, -1.09]]
    base = [[0, 0, 0]] + hs
    perm = [[0, 0, 0]] + [hs[3], hs[0], hs[2], hs[1]]
    a = ("a", _ads(["C"] + ["H"] * 4, [[5 + x, 5 + y, 12 + z] for x, y, z in base]))
    b = ("b", _ads(["C"] + ["H"] * 4, [[5 + x, 5 + y, 12 + z] for x, y, z in perm]))
    assert len(filter_periodic_sites([a, b], CELL, PBC, 5,
                                     com_tol_ang=0.4, orient_tol_ang=0.3)) == 1


def test_periodic_filter_keeps_lowest_energy_representative():
    from beep.core.periodic_sampler import filter_periodic_sites

    a = ("high", _ads(["C", "O"], [[5.0, 5.0, 12.0], [5.0, 5.0, 13.13]]))
    b = ("low", _ads(["C", "O"], [[5.1, 5.0, 12.0], [5.1, 5.0, 13.13]]))
    unique = filter_periodic_sites([a, b], CELL, PBC, 2, com_tol_ang=1.0,
                                   energies={"high": -10.0, "low": -11.0})
    assert [n for n, _ in unique] == ["low"]


def test_site_filter_config_defaults_to_periodic():
    from beep.models.sampling_periodic import SamplingPeriodicConfig

    cfg = SamplingPeriodicConfig(**_periodic_config_kwargs())
    assert cfg.site_filter == "periodic"
    assert cfg.orientation_tol_ang == 0.3


# ---------------------------------------------------------------------------
# Regressions: seeded rotations, face-straddling adsorbates, spin bookkeeping
# ---------------------------------------------------------------------------

_CAND_KW = dict(
    z_top_bohr=5.0 * ANG2BOHR,
    z_scan_range_bohr=(0.0, 5.0 * ANG2BOHR),
    sampling_distance_bohr=2.5 * ANG2BOHR,
    cavity_scan_step_bohr=0.5 * ANG2BOHR,
    cavity_window_bohr=1.0 * ANG2BOHR,
    sanity_min_dist_bohr=1.5 * ANG2BOHR,
    sanity_max_iter=20,
)


def _candidate(seed, x_ang=2.0, y_ang=2.0):
    cell_diag = np.array([10.0, 10.0, 30.0])
    result = generate_candidate(
        _tiny_slab(cell_diag), _tiny_adsorbate(),
        x_bohr=x_ang * ANG2BOHR, y_bohr=y_ang * ANG2BOHR,
        cell_diag_bohr=cell_diag, pbc=[True, True, False],
        rng=random.Random(seed), **_CAND_KW,
    )
    assert result is not None
    return result


def test_random_seed_controls_adsorbate_rotation():
    """Regression: the rotation came from numpy's global RNG (qcelemental's scramble),
    so seeding Python's `random` reproduced the grid noise but not the orientation."""
    np.random.seed(1)
    mol_a, _ = _candidate(7)
    np.random.seed(2)          # must be irrelevant now
    mol_b, _ = _candidate(7)
    np.testing.assert_allclose(mol_a.geometry, mol_b.geometry)


def test_different_seeds_give_different_rotations():
    mol_a, _ = _candidate(7)
    mol_b, _ = _candidate(8)
    assert not np.allclose(mol_a.geometry[3:], mol_b.geometry[3:])


def test_rotation_does_not_touch_global_random_state():
    """The old code reseeded Python's global `random` on every attempt."""
    random.seed(123)
    expected = random.random()
    random.seed(123)
    _candidate(7)
    assert random.random() == expected


def test_recenter_adsorbate_straddling_cell_face():
    """An adsorbate wrapped per atom across x=0 must be recentred as one molecule:
    the naive mean of the split coordinates sat mid-cell, so the 'centered' entry
    kept a split adsorbate."""
    L = 10.0 * ANG2BOHR
    cell_diag = np.array([L, L, 30.0 * ANG2BOHR])
    pbc = [True, True, False]
    bond = 1.13 * ANG2BOHR
    geom = np.array([
        [5.0 * ANG2BOHR, 5.0 * ANG2BOHR, 0.0],       # slab atom
        [L - 0.3 * ANG2BOHR, 5.0 * ANG2BOHR, 5.0],   # C, wrapped to the far face
        [-0.3 * ANG2BOHR + bond, 5.0 * ANG2BOHR, 5.0],  # O, just inside x=0
    ])
    out = recenter_adsorbate_com(geom, n_surface_atoms=1, cell_diag_bohr=cell_diag, pbc=pbc)
    c, o = out[1], out[2]
    # contiguous: the plain Euclidean bond length equals the true one
    assert np.linalg.norm(o - c) == pytest.approx(bond)
    # and centred: the (unweighted) adsorbate centre is at Lx/2, Ly/2
    assert 0.5 * (c[0] + o[0]) == pytest.approx(0.5 * L)
    assert 0.5 * (c[1] + o[1]) == pytest.approx(0.5 * L)


def test_generate_candidate_on_face_keeps_adsorbate_contiguous():
    """Grid nodes on the x=0 / y=0 lines rotate the adsorbate partly to negative x.
    The per-atom wrap used to split it before recentering."""
    bond = 1.13 * ANG2BOHR
    for seed in range(6):
        mol, _ = _candidate(seed, x_ang=0.0, y_ang=2.0)
        ads = mol.geometry[3:]
        assert np.linalg.norm(ads[1] - ads[0]) == pytest.approx(bond, rel=1e-6)
        assert ads[:, 0].mean() == pytest.approx(5.0)   # cell is 10 bohr wide


def test_periodic_filter_com_of_wrapped_adsorbate():
    """Duplicate filtering must see a split adsorbate's true COM (on the molecule),
    not the naive mean in the middle of the cell."""
    from beep.core.periodic_sampler import filter_periodic_sites

    # same CO, once contiguous near x=0.3 A, once stored wrapped across x=0
    a = ("a", _ads(["C", "O"], [[0.3, 5.0, 12.0], [0.3, 5.0, 13.13]]))
    split = ("split", _ads(["C", "O"], [[30.8, 5.0, 12.0], [0.9, 5.0, 13.13]]))
    # rotated CO lying along x, straddling the face: C at x=-0.2 -> 30.8, O at +0.9
    assert len(filter_periodic_sites([a, split], CELL, PBC, 2, com_tol_ang=1.0,
                                     orient_tol_ang=None)) == 1


def _doublet_hco():
    return qcel.models.Molecule(
        symbols=["C", "O", "H"],
        geometry=np.array([[-0.62, 0.04, 0.0], [0.53, -0.10, 0.0], [-1.15, 1.03, 0.0]]) * ANG2BOHR,
        molecular_charge=0, molecular_multiplicity=2,
        fix_com=False, fix_orientation=False,
    )


def test_combine_propagates_adsorbate_multiplicity_and_fragments():
    """An open-shell adsorbate must keep its spin state in the stored complex."""
    from beep.core.periodic_sampler import _combine

    cell_diag = np.array([10.0, 10.0, 30.0])
    slab = _tiny_slab(cell_diag)
    hco = _doublet_hco()
    ads_coords = hco.geometry + np.array([5.0 * ANG2BOHR, 5.0 * ANG2BOHR, 3.0 * ANG2BOHR])
    mol = _combine(slab, hco, ads_coords)
    assert mol.molecular_multiplicity == 2
    assert mol.molecular_charge == pytest.approx(0.0)
    assert [list(f) for f in mol.fragments] == [[0, 1, 2], [3, 4, 5]]
    assert list(mol.fragment_multiplicities) == [1, 2]
    assert list(mol.fragment_charges) == [0.0, 0.0]


def test_generate_candidate_keeps_doublet_adsorbate():
    cell_diag = np.array([10.0, 10.0, 30.0])
    result = generate_candidate(
        _tiny_slab(cell_diag), _doublet_hco(),
        x_bohr=2.0 * ANG2BOHR, y_bohr=2.0 * ANG2BOHR,
        cell_diag_bohr=cell_diag, pbc=[True, True, False],
        rng=random.Random(0), **_CAND_KW,
    )
    assert result is not None
    mol, _ = result
    assert mol.molecular_multiplicity == 2
    assert list(mol.fragment_multiplicities) == [1, 2]


def test_strip_adsorbate_keeps_surface_state_from_fragments():
    from beep.core.periodic_sampler import _combine

    cell_diag = np.array([10.0, 10.0, 30.0])
    slab = _tiny_slab(cell_diag)
    hco = _doublet_hco()
    ads_coords = hco.geometry + np.array([5.0 * ANG2BOHR, 5.0 * ANG2BOHR, 3.0 * ANG2BOHR])
    bare = strip_adsorbate(_combine(slab, hco, ads_coords), n_surface_atoms=3)
    assert list(bare.symbols) == ["O", "O", "O"]
    assert bare.molecular_multiplicity == 1
    assert bare.molecular_charge == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Workflow: adsorbate comes from the entry's initial-molecule slot
# ---------------------------------------------------------------------------

def test_periodic_workflow_reads_adsorbate_from_entry_slot(tmp_path, monkeypatch):
    """Regression: the adsorbate lookup went through fetch_opt_record, which needs an
    optimization record at the MACE spec; an MLP-only run has none, so every run
    fell through to the atoms collection (or died). The slab side already used
    fetch_entry_initial_molecule; the adsorbate must too."""
    from unittest.mock import MagicMock, patch
    from beep.models.sampling_periodic import SamplingPeriodicConfig
    from beep.workflows import sampling_periodic

    monkeypatch.chdir(tmp_path)
    cfg = SamplingPeriodicConfig(**_periodic_config_kwargs())

    ds_sm = MagicMock(); ds_sm.entry_names = ["CO", "H2O"]
    ds_surf = MagicMock(); ds_surf.entry_names = []      # no slabs -> loop body skipped

    def get_collection(client, kind, name):
        return ds_sm if name == cfg.small_molecule_collection else ds_surf

    with patch.object(sampling_periodic, "qcf") as qcf:
        qcf.get_collection.side_effect = get_collection
        sampling_periodic.run(cfg, MagicMock())

    qcf.fetch_entry_initial_molecule.assert_called_once_with(ds_sm, "CO")
    qcf.fetch_initial_molecule.assert_not_called()
    qcf.fetch_opt_record.assert_not_called()
    qcf.fetch_atom_molecule.assert_not_called()


def test_periodic_workflow_falls_back_to_atoms_collection(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch
    from beep.models.sampling_periodic import SamplingPeriodicConfig
    from beep.workflows import sampling_periodic

    monkeypatch.chdir(tmp_path)
    cfg = SamplingPeriodicConfig(**_periodic_config_kwargs(molecule="H"))
    ds_sm = MagicMock(); ds_sm.entry_names = ["CO"]
    ds_surf = MagicMock(); ds_surf.entry_names = []

    with patch.object(sampling_periodic, "qcf") as qcf:
        qcf.get_collection.side_effect = lambda c, k, n: ds_sm if n == cfg.small_molecule_collection else ds_surf
        client = MagicMock()
        sampling_periodic.run(cfg, client)

    qcf.fetch_atom_molecule.assert_called_once_with(client, cfg.atoms_collection, "H")
    qcf.fetch_entry_initial_molecule.assert_not_called()
