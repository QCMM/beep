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
from typing import List, Optional, Sequence, Tuple

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

def _atom_index_ranges(indices_0based: Sequence[int]) -> str:
    """Compress a sorted 0-indexed list to a geomeTRIC 1-based range spec (`1-4,7,9-10`)."""
    ones = sorted(set(int(i) + 1 for i in indices_0based))
    parts: List[str] = []
    start = end = ones[0]
    for i in ones[1:]:
        if i == end + 1:
            end = i
        else:
            parts.append(f"{start}-{end}" if start != end else str(start))
            start = end = i
    parts.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(parts)


def build_freeze_constraint_string(indices_0based: Sequence[int]) -> Optional[str]:
    """Return a geomeTRIC ``constraints`` block freezing the given 0-indexed atoms.

    geomeTRIC uses 1-based indices in its constraints file format; this helper
    converts and range-compresses. Returns ``None`` if the list is empty.
    """
    if not indices_0based:
        return None
    return f"$freeze\nxyz {_atom_index_ranges(indices_0based)}\n"


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
) -> Optional[Molecule]:
    """Try to build one valid slab+adsorbate structure at the given (x, y).

    Returns a fully-formed qcelemental Molecule (surface atoms first, adsorbate
    atoms last) with adsorbate coordinates wrapped into the cell, or None if
    no valid placement was found within ``sanity_max_iter`` rotation attempts.
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
            return _combine(surface, adsorbate, ads_coords)
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
            mol = generate_candidate(
                surface, adsorbate, x, y, z_top, z_scan_range,
                sampling_distance_bohr, cell_diag_bohr, pbc,
                cavity_scan_step_bohr, cavity_window_bohr,
                sanity_min_dist_bohr, sanity_max_iter, rng,
            )
            name = f"X{ix:02d}_Y{iy:02d}"
            if mol is None:
                logger.info(
                    f"  skip {name}: no valid placement at "
                    f"({x*BOHR2ANG:.2f}, {y*BOHR2ANG:.2f}) A"
                )
                continue
            candidates.append((name, mol))
            all_ads_coords.append(mol.geometry[len(surface.symbols):].reshape(-1, 3))

    # Build a single debug molecule for visualisation
    if all_ads_coords:
        debug_symbols = list(surface.symbols) + list(adsorbate.symbols) * len(all_ads_coords)
        debug_geom = np.concatenate(
            [surface_geom.flatten()] + [c.flatten() for c in all_ads_coords]
        )
    else:
        debug_symbols = list(surface.symbols)
        debug_geom = surface_geom.flatten()
    debug_mol = qcel.models.Molecule(
        symbols=debug_symbols, geometry=debug_geom, fix_com=False, fix_orientation=False,
    )

    logger.info(
        f"Periodic sampler: generated {len(candidates)} valid candidates "
        f"out of {n_x * n_y} grid nodes."
    )
    return candidates, debug_mol
