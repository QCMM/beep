"""Guard against silently reusing a dataset entry for a different geometry.

QCFractal dataset entries are keyed by name, and BEEP's periodic workflows name them by
binding site. A later run that produces the same names with different geometries (a
resampling, another sampling model, another set of optimized complexes) would otherwise
reuse the old entries and compute the new level of theory on the OLD geometries.
Runs with different geometries belong in different datasets (``dataset_suffix``).
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

BOHR2ANG = 0.529177210903


def _deviation_bohr(a, b, cell_ang=None, pbc=None):
    """Largest per-coordinate deviation; along periodic axes modulo the lattice vector
    (orthorhombic cell), so a copy wrapped into another periodic image counts as equal."""
    d = (np.asarray(a, dtype=float).reshape(-1, 3) - np.asarray(b, dtype=float).reshape(-1, 3))
    if cell_ang is not None and pbc is not None:
        lengths = np.diag(np.asarray(cell_ang, dtype=float)) / BOHR2ANG
        for k in range(3):
            if pbc[k] and lengths[k] > 0:
                d[:, k] -= lengths[k] * np.round(d[:, k] / lengths[k])
    return float(np.abs(d).max()) if d.size else 0.0


def check_entry_geometry(existing_mol, new_mol, entry_name: str, dataset_name: str,
                         tol_bohr: float = 1e-4, cell_ang=None, pbc=None) -> None:
    a = np.asarray(existing_mol.geometry, dtype=float).ravel()
    b = np.asarray(new_mol.geometry, dtype=float).ravel()
    same_atoms = list(existing_mol.symbols) == list(new_mol.symbols)
    dev_bohr = _deviation_bohr(a, b, cell_ang, pbc) if same_atoms and a.shape == b.shape else None
    if dev_bohr is None or dev_bohr > tol_bohr:
        dev = f"{dev_bohr * BOHR2ANG:.3f} A" if dev_bohr is not None else "different atoms"
        raise ValueError(
            f"{dataset_name}/{entry_name}: an entry of this name already exists with a "
            f"different geometry (max deviation {dev}). Entries keep the geometry of the run "
            f"that created them, so this run would silently compute on the old geometry. "
            f"Give this run its own datasets with 'dataset_suffix' (e.g. '_v1')."
        )


def guard_reused_entries(ds, pairs: Iterable[Tuple[str, object]], optimization: bool,
                         cell_ang=None, pbc=None) -> None:
    """Check every (name, molecule) whose name already exists in ``ds``. For periodic
    systems pass the cell and pbc: geometries are then compared modulo lattice vectors."""
    wanted = {name: mol for name, mol in pairs}
    reused = [n for n in ds.entry_names if n in wanted]
    if not reused:
        return
    for entry in ds.iterate_entries(entry_names=reused):
        existing = entry.initial_molecule if optimization else entry.molecule
        check_entry_geometry(existing, wanted[entry.name], entry.name, ds.name, cell_ang=cell_ang, pbc=pbc)
