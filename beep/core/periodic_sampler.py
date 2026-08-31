"""Grid-based sampler for adsorbate placements on a periodic slab.

Ported from gbovolenta's ``sampling_grid_noise_*.py`` monoliths, with:

- Minimum-image distances and cell-wrap under periodic boundary conditions
- Full-footprint grid (no 1 A dead border on each side)
- Widened, configurable cavity z-scan window; best-fit z instead of first-hit
- Graceful skip when a grid point admits no valid placement
- Sanity check over every adsorbate atom (not just one)
- Iteration cap on the sanity-check retry loop
- Seeded RNG for reproducibility

Assumes an orthogonal cell (typical for slab supercells). Non-orthogonal
cells would need the fractional-coordinate inverse-cell transformation
in `min_image_vec` / `wrap_into_cell`.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import qcelemental as qcel
from qcelemental.models import Molecule

BOHR2ANG = qcel.constants.conversion_factor("bohr", "angstrom")
ANG2BOHR = qcel.constants.conversion_factor("angstrom", "bohr")


# ---------------------------------------------------------------------------
# PBC utilities (orthogonal cell)
# ---------------------------------------------------------------------------

def _cell_diag_bohr(cell_ang: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the diagonal of an orthogonal 3x3 cell (Angstrom) in Bohr."""
    cell = np.asarray(cell_ang, dtype=float)
    return np.array([cell[0, 0], cell[1, 1], cell[2, 2]]) * ANG2BOHR


def min_image_vec(dp: np.ndarray, cell_diag_bohr: np.ndarray, pbc: Sequence[bool]) -> np.ndarray:
    """Shortest displacement `dp` (Bohr) under orthogonal-cell periodicity."""
    out = np.asarray(dp, dtype=float).copy()
    for i in range(3):
        if pbc[i] and cell_diag_bohr[i] > 0:
            L = cell_diag_bohr[i]
            out[i] -= L * np.round(out[i] / L)
    return out


def min_image_distance(
    p1: np.ndarray, p2: np.ndarray, cell_diag_bohr: np.ndarray, pbc: Sequence[bool]
) -> float:
    """Minimum-image distance between two points (Bohr)."""
    return float(np.linalg.norm(min_image_vec(np.asarray(p2) - np.asarray(p1), cell_diag_bohr, pbc)))


def wrap_into_cell(
    coords: np.ndarray, cell_diag_bohr: np.ndarray, pbc: Sequence[bool]
) -> np.ndarray:
    """Wrap atomic coordinates (Bohr, shape (N,3)) into the primary cell."""
    out = np.asarray(coords, dtype=float).copy()
    for i in range(3):
        if pbc[i] and cell_diag_bohr[i] > 0:
            L = cell_diag_bohr[i]
            out[:, i] -= L * np.floor(out[:, i] / L)
    return out


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_grid(
    cell_diag_bohr: np.ndarray,
    step_size_bohr: float,
    noise_frac: float,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_grid, y_grid) in Bohr covering the full [0, Lx) x [0, Ly) footprint.

    Interior nodes get an independent random jitter of ±noise_frac * step_size.
    Boundary nodes (0 and Lx - step) are left unperturbed so PBC wrapping of
    the two ends stays consistent.
    """
    Lx, Ly = cell_diag_bohr[0], cell_diag_bohr[1]
    x_grid = np.arange(0.0, Lx, step_size_bohr, dtype=float)
    y_grid = np.arange(0.0, Ly, step_size_bohr, dtype=float)
    if noise_frac > 0.0:
        span = noise_frac * step_size_bohr
        for i in range(1, len(x_grid) - 1):
            x_grid[i] += rng.uniform(-span, span)
        for i in range(1, len(y_grid) - 1):
            y_grid[i] += rng.uniform(-span, span)
    return x_grid, y_grid


# ---------------------------------------------------------------------------
# Placement primitives
# ---------------------------------------------------------------------------

def nearest_surface_atom(
    surface_geom_bohr: np.ndarray,
    point_bohr: np.ndarray,
    cell_diag_bohr: np.ndarray,
    pbc: Sequence[bool],
) -> Tuple[int, float]:
    """Return (atom_index, min_image_distance_bohr) of the surface atom closest to `point`."""
    diffs = np.array(
        [min_image_vec(point_bohr - a, cell_diag_bohr, pbc) for a in surface_geom_bohr]
    )
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def hemisphere_z_shift(sampling_distance_bohr: float, xy_dist_bohr: float) -> Optional[float]:
    """Height above the nearest surface atom placing the adsorbate at ``sampling_distance``.

    Returns None if ``xy_dist > sampling_distance`` (no real hemisphere solution).
    """
    r2 = sampling_distance_bohr ** 2 - xy_dist_bohr ** 2
    if r2 < 0:
        return None
    return float(np.sqrt(r2))


def find_cavity_z(
    surface_geom_bohr: np.ndarray,
    x: float,
    y: float,
    z_range_bohr: Tuple[float, float],
    scan_step_bohr: float,
    sampling_distance_bohr: float,
    window_bohr: float,
    cell_diag_bohr: np.ndarray,
    pbc: Sequence[bool],
) -> Optional[float]:
    """Scan z to find the best-fit height inside a cavity.

    Returns the z (Bohr) at which the nearest-atom distance is closest to
    ``sampling_distance`` while still within
    ``[sampling_distance - window, sampling_distance]``, or None if no
    z in the scan range qualifies. Scans top-down; ties broken by first hit.
    """
    z_lo, z_hi = z_range_bohr
    z_grid = np.arange(z_lo, z_hi, scan_step_bohr)
    best_z: Optional[float] = None
    best_err = float("inf")
    lower = sampling_distance_bohr - window_bohr
    upper = sampling_distance_bohr
    for z in reversed(z_grid):
        _, d = nearest_surface_atom(
            surface_geom_bohr, np.array([x, y, z]), cell_diag_bohr, pbc
        )
        if lower <= d <= upper:
            err = abs(d - sampling_distance_bohr)
            if err < best_err:
                best_err = err
                best_z = float(z)
    return best_z


def all_atoms_ok(
    ads_coords_bohr: np.ndarray,
    surface_geom_bohr: np.ndarray,
    cell_diag_bohr: np.ndarray,
    pbc: Sequence[bool],
    min_dist_bohr: float,
) -> bool:
    """True if every adsorbate atom is at least ``min_dist`` from every surface atom."""
    for a in ads_coords_bohr:
        for s in surface_geom_bohr:
            if min_image_distance(a, s, cell_diag_bohr, pbc) < min_dist_bohr:
                return False
    return True


# ---------------------------------------------------------------------------
# Freeze constraints for geomeTRIC
# ---------------------------------------------------------------------------

def build_freeze_constraints(indices_0based: Sequence[int]) -> Optional[Dict[str, Any]]:
    """Return a geomeTRIC ``constraints`` dict freezing the given 0-indexed atoms.

    geomeTRIC's JSON API (``geometric.run_json``, the path QCEngine drives) takes
    the *structured* constraints form::

        {"freeze": [{"type": "xyz", "indices": [0, 1, 2]}]}

    and renders the classic ``$freeze / xyz 1-3`` text itself. Handing it the
    pre-rendered text instead makes it raise
    ``AttributeError: 'str' object has no attribute 'items'`` before the first
    gradient, so every frozen-slab optimization dies on submission.

    Indices stay **0-based** here: geomeTRIC's ``commadash`` does the 0->1 shift
    and the range compression (``[0,1,2] -> "1-3"``). Returns ``None`` if the
    list is empty.
    """
    if not indices_0based:
        return None
    ordered = sorted(set(int(i) for i in indices_0based))
    return {"freeze": [{"type": "xyz", "indices": ordered}]}


def write_overlay_xyz(path, symbols: Sequence[str], geom_bohr: np.ndarray) -> None:
    """Write the sampling-coverage overlay (slab + all accepted adsorbate copies).

    Plain XYZ writer: the overlay intentionally contains overlapping copies, so it
    must not be routed through qcelemental's physical-geometry validation.
    """
    geom = np.asarray(geom_bohr, dtype=float).reshape(-1, 3) * BOHR2ANG
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\nsampling coverage overlay (slab + accepted adsorbate copies)\n")
        for sym, (x, y, z) in zip(symbols, geom):
            fh.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")


def frozen_atom_indices(
    surface_geom_bohr: np.ndarray,
    freeze_below_z_ang: Optional[float],
    freeze_atoms: Optional[Sequence[int]],
    n_surface_atoms: int,
) -> List[int]:
    """Compute the 0-indexed freeze list for the combined slab+adsorbate molecule.

    ``freeze_atoms`` (explicit list) overrides ``freeze_below_z_ang`` (z threshold
    applied only to slab atoms; adsorbate atoms are never frozen automatically).
    """
    if freeze_atoms is not None:
        return list(freeze_atoms)
    if freeze_below_z_ang is not None:
        z_bohr = freeze_below_z_ang * ANG2BOHR
        return [i for i in range(n_surface_atoms) if surface_geom_bohr[i, 2] < z_bohr]
    return []


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidate(
    surface: Molecule,
    adsorbate: Molecule,
    x_bohr: float,
    y_bohr: float,
    z_top_bohr: float,
    z_scan_range_bohr: Tuple[float, float],
    sampling_distance_bohr: float,
    cell_diag_bohr: np.ndarray,
    pbc: Sequence[bool],
    cavity_scan_step_bohr: float,
    cavity_window_bohr: float,
    sanity_min_dist_bohr: float,
    sanity_max_iter: int,
    rng: random.Random,
) -> Optional[Tuple[Molecule, np.ndarray]]:
    """Try to build one valid slab+adsorbate structure at the given (x, y).

    Returns ``(centered_mol, original_ads_coords)`` on success, or ``None``
    if no valid placement was found within ``sanity_max_iter`` rotation
    attempts. The returned ``centered_mol`` has been slid in xy so the
    adsorbate COM sits at the periodic cell center (pure gauge shift; use
    this for compute + per-candidate xyz files). ``original_ads_coords`` is
    the adsorbate placement *before* the recentering shift (used by the
    aggregate debug xyz so the sampling coverage remains visible instead of
    collapsing to a single point at cell center).
    """
    surface_geom = surface.geometry
    idx, xy_atom_dist = nearest_surface_atom(
        surface_geom, np.array([x_bohr, y_bohr, z_top_bohr]), cell_diag_bohr, pbc
    )
    nearest_xyz = surface_geom[idx]
    xy_dist_to_nearest_atom = min_image_distance(
        np.array([x_bohr, y_bohr, nearest_xyz[2]]), nearest_xyz, cell_diag_bohr, pbc
    )

    if xy_dist_to_nearest_atom <= sampling_distance_bohr:
        # Hemisphere path: place on the sphere of radius sampling_distance
        # centered on the nearest surface atom.
        dz = hemisphere_z_shift(sampling_distance_bohr, xy_dist_to_nearest_atom)
        if dz is None:
            return None
        shift_vect = np.array([x_bohr, y_bohr, nearest_xyz[2] + dz])
    else:
        # Cavity path: z-scan for a good height.
        z_shift = find_cavity_z(
            surface_geom, x_bohr, y_bohr, z_scan_range_bohr,
            cavity_scan_step_bohr, sampling_distance_bohr, cavity_window_bohr,
            cell_diag_bohr, pbc,
        )
        if z_shift is None:
            return None
        shift_vect = np.array([x_bohr, y_bohr, z_shift])

    # Try random rotations until the sanity check passes.
    for _ in range(sanity_max_iter):
        # qcelemental's Molecule.scramble draws from Python's global `random`;
        # seed it fresh from `rng` on every attempt so results are reproducible
        # given the same top-level seed.
        random.seed(rng.randrange(2 ** 31 - 1))
        mol_shifted = adsorbate.scramble(
            do_shift=shift_vect, do_rotate=True, do_resort=False, deflection=1.0
        )[0]
        ads_coords = wrap_into_cell(mol_shifted.geometry, cell_diag_bohr, pbc)
        if all_atoms_ok(ads_coords, surface_geom, cell_diag_bohr, pbc, sanity_min_dist_bohr):
            # Return the CENTERED combined molecule (used for compute + saved
            # per-candidate xyz) plus the pre-shift adsorbate coordinates
            # (used by the aggregate debug xyz so the sampling coverage stays
            # visible instead of collapsing to a single point at cell center).
            combined = _combine(surface, adsorbate, ads_coords)
            n_surf = len(surface.symbols)
            centered_geom = recenter_adsorbate_com(
                combined.geometry, n_surf, cell_diag_bohr, pbc
            )
            centered_mol = qcel.models.Molecule(
                symbols=list(combined.symbols),
                geometry=centered_geom.flatten(),
                fix_com=False,
                fix_orientation=False,
            )
            return centered_mol, ads_coords
    return None


def _combine(surface: Molecule, adsorbate: Molecule, ads_coords_bohr: np.ndarray) -> Molecule:
    """Build a combined qcel Molecule from a slab and adsorbate at chosen coords."""
    symbols = list(surface.symbols) + list(adsorbate.symbols)
    geometry = np.concatenate([surface.geometry.flatten(), ads_coords_bohr.flatten()])
    return qcel.models.Molecule(
        symbols=symbols,
        geometry=geometry,
        fix_com=False,
        fix_orientation=False,
    )


def strip_adsorbate(
    combined_mol: Molecule, n_surface_atoms: int
) -> Molecule:
    """Return a qcel Molecule of just the surface atoms from a combined slab+adsorbate.

    Assumes the ``_combine()`` convention (surface first, adsorbate last).
    Used to build the per-site bare-surface starting geometry for the
    ``_surface`` sibling OptimizationDataset: the surface positions after the
    complex is optimized carry the site-specific deformation, and re-relaxing
    from that state gives a physically clean bare-surface reference for BE.
    """
    n = int(n_surface_atoms)
    symbols = list(combined_mol.symbols[:n])
    geometry = np.asarray(combined_mol.geometry, dtype=float).reshape(-1, 3)[:n]
    return qcel.models.Molecule(
        symbols=symbols,
        geometry=geometry.flatten(),
        fix_com=False,
        fix_orientation=False,
    )


def recenter_adsorbate_com(
    combined_geom: np.ndarray,
    n_surface_atoms: int,
    cell_diag_bohr: np.ndarray,
    pbc: Sequence[bool],
) -> np.ndarray:
    """Slide every atom in xy so the adsorbate COM lands at the cell center.

    Pure PBC gauge shift — energy, gradient, Hessian invariant. Only the
    periodic axes (x, y in a standard slab; per `pbc`) are shifted; z is
    left unchanged so `freeze_below_z_ang` still picks the same atoms and
    the vacuum gap is preserved. Returns wrapped coordinates.
    """
    coords = np.asarray(combined_geom, dtype=float).copy().reshape(-1, 3)
    ads_com = coords[n_surface_atoms:].mean(axis=0)
    target = 0.5 * cell_diag_bohr
    shift = np.zeros(3)
    for i in range(3):
        if pbc[i]:
            shift[i] = target[i] - ads_com[i]
    coords += shift
    return wrap_into_cell(coords, cell_diag_bohr, pbc)


def run_periodic_sampling(
    surface: Molecule,
    adsorbate: Molecule,
    cell_ang: Sequence[Sequence[float]],
    pbc: Sequence[bool],
    step_size_ang: float,
    grid_noise_frac: float,
    sampling_distance_ang: float,
    cavity_z_scan_step_ang: float,
    cavity_z_scan_window_ang: float,
    sanity_min_distance_ang: float,
    sanity_max_iter: int,
    rng: random.Random,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[str, Molecule]], Molecule]:
    """Generate one candidate per grid node over the slab's periodic footprint.

    Returns ``(candidates, debug_molecule)`` where:
    - ``candidates``: list of ``(entry_name, qcel.Molecule)`` — one entry per
      grid node that produced a valid placement. Skipped nodes are logged.
    - ``debug_molecule``: a single Molecule containing the slab plus every
      accepted adsorbate copy, useful for a-glance visualisation.
    """
    logger = logger or logging.getLogger(__name__)
    cell_diag_bohr = _cell_diag_bohr(cell_ang)

    step_size_bohr = step_size_ang * ANG2BOHR
    sampling_distance_bohr = sampling_distance_ang * ANG2BOHR
    cavity_scan_step_bohr = cavity_z_scan_step_ang * ANG2BOHR
    cavity_window_bohr = cavity_z_scan_window_ang * ANG2BOHR
    sanity_min_dist_bohr = sanity_min_distance_ang * ANG2BOHR

    x_grid, y_grid = build_grid(cell_diag_bohr, step_size_bohr, grid_noise_frac, rng)

    # z reference: top of the slab plus a bit of clearance for the initial
    # nearest-atom probe. Use the slab's max z + sampling_distance so cavity
    # detection has a sensible "above the surface" starting point.
    surface_geom = surface.geometry
    z_top = float(surface_geom[:, 2].max()) + sampling_distance_bohr
    z_scan_range = (0.0, float(surface_geom[:, 2].max()) + sampling_distance_bohr)

    candidates: List[Tuple[str, Molecule]] = []
    all_ads_coords: List[np.ndarray] = []

    n_x, n_y = len(x_grid), len(y_grid)
    logger.info(
        f"Periodic sampler: {n_x} x {n_y} grid over cell "
        f"{cell_diag_bohr[0]*BOHR2ANG:.2f} x {cell_diag_bohr[1]*BOHR2ANG:.2f} A "
        f"(step {step_size_ang} A, noise ±{grid_noise_frac*step_size_ang:.2f} A)"
    )

    for ix, x in enumerate(x_grid):
        for iy, y in enumerate(y_grid):
            result = generate_candidate(
                surface, adsorbate, x, y, z_top, z_scan_range,
                sampling_distance_bohr, cell_diag_bohr, pbc,
                cavity_scan_step_bohr, cavity_window_bohr,
                sanity_min_dist_bohr, sanity_max_iter, rng,
            )
            name = f"X{ix:02d}_Y{iy:02d}"
            if result is None:
                logger.info(
                    f"  skip {name}: no valid placement at "
                    f"({x*BOHR2ANG:.2f}, {y*BOHR2ANG:.2f}) A"
                )
                continue
            centered_mol, orig_ads_coords = result
            candidates.append((name, centered_mol))
            all_ads_coords.append(orig_ads_coords)

    # Build a single debug molecule for visualisation
    if all_ads_coords:
        debug_symbols = list(surface.symbols) + list(adsorbate.symbols) * len(all_ads_coords)
        debug_geom = np.concatenate(
            [surface_geom.flatten()] + [c.flatten() for c in all_ads_coords]
        )
    else:
        debug_symbols = list(surface.symbols)
        debug_geom = surface_geom.flatten()
    # NOTE: deliberately NOT a qcel Molecule. This is a visualisation OVERLAY (the slab
    # plus every accepted adsorbate copy at its pre-centering placement), not a physical
    # system: independent grid nodes routinely drop copies a fraction of an Angstrom
    # apart, which trips qcelemental's "atoms are too close" validator and used to abort
    # the entire sampling run over a cosmetic artifact. qcelemental exposes no way to
    # relax that check (``validate=False`` skips schema-filling and then breaks
    # ``to_file``; ``nonphysical=True`` only covers masses/charges), so the overlay is
    # returned as raw arrays and written by :func:`write_overlay_xyz`.
    overlay = (debug_symbols, debug_geom)

    logger.info(
        f"Periodic sampler: generated {len(candidates)} valid candidates "
        f"out of {n_x * n_y} grid nodes."
    )
    return candidates, overlay
