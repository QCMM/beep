"""
Pure sampling logic — shell generation, RMSD filtering, binding site deduplication.

No QCFractal imports. Only uses qcelemental.models.Molecule as a data type.
"""
import logging
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
from qcelemental.models.molecule import Molecule


def generate_shell_list(sampling_shell: float, condition: str) -> List[float]:
    """
    Generate a list of sampling shells based on the given condition.

    Parameters:
    - sampling_shell (float): The primary sampling shell value.
    - condition (str): The condition to adjust the sampling shell list.
                       It can be one of 'sparse', 'normal', or 'fine'.

    Returns:
    - List[float]: A list containing the adjusted sampling shell values.

    Raises:
    - ValueError: If the condition is not one of ['sparse', 'normal', 'fine'].

    Examples:
    >>> generate_shell_list(10.0, 'sparse')
    [10.0]

    >>> generate_shell_list(10.0, 'normal')
    [10.0, 8.0, 12.0]

    >>> generate_shell_list(10.0, 'fine')
    [10.0, 7.5, 9.0, 11.0, 15.0]
    """

    conditions_map = {
        "sparse": [sampling_shell],
        "normal": [sampling_shell, sampling_shell * 0.8, sampling_shell * 1.2],
        "fine": [
            sampling_shell,
            sampling_shell * 0.8,
            sampling_shell * 1.2,
            sampling_shell * 0.75,
            sampling_shell * 1.5,
        ],
        "hyperfine": [
            sampling_shell,
            sampling_shell * 0.8,
            sampling_shell * 1.2,
            sampling_shell * 0.75,
            sampling_shell * 1.5,
            sampling_shell * 0.9,
            sampling_shell * 1.1,
        ],
    }

    shell_list = conditions_map.get(condition)
    if shell_list is None:
        raise ValueError(
            "Condition should be one of ['sparse', 'normal', 'fine', 'hyperfine']"
        )

    return shell_list


def _lig_block(mol, L: int) -> np.ndarray:
    return np.asarray(mol.geometry)[-L:]

def _key_for_grid(mol, L: int, grid: float) -> Tuple[int, int, int]:
    com = _lig_block(mol, L).mean(axis=0)
    return tuple((com / grid).astype(int))

def _neighbor_keys(key: Tuple[int, int, int], radius: int):
    ix, iy, iz = key
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                yield (ix + dx, iy + dy, iz + dz)

def _cheb_within(k1: Tuple[int,int,int], k2: Tuple[int,int,int], r: int) -> bool:
    return (abs(k1[0]-k2[0]) <= r and
            abs(k1[1]-k2[1]) <= r and
            abs(k1[2]-k2[2]) <= r)

def ligand_pairdist_ok(m1, m2, L: int, tau: float = 1e-3) -> bool:
    """
    Permutation- & rigid-motion-invariant ligand gate:
    compare sorted upper-triangle pairwise distances.
    """
    X1 = _lig_block(m1, L); X2 = _lig_block(m2, L)
    D1 = np.linalg.norm(X1[:, None, :] - X1[None, :, :], axis=2)
    D2 = np.linalg.norm(X2[:, None, :] - X2[None, :, :], axis=2)
    iu = np.triu_indices(L, 1)
    v1 = np.sort(D1[iu]); v2 = np.sort(D2[iu])
    rms = float(np.sqrt(np.mean((v1 - v2)**2)))
    return rms <= tau

def compute_rmsd_conditional(m1, m2, rmsd_symm: bool, cutoff: float, atoms_map: bool = True) -> Tuple[float, float]:
    r = m1.align(m2, atoms_map=atoms_map)[1]["rmsd"]
    if rmsd_symm and r >= cutoff:
        rm = m1.align(m2, atoms_map=atoms_map, run_mirror=True)[1]["rmsd"]
    else:
        rm = 10.0
    return r, rm

# ---------- main filter (OR gating + fallback) ----------

def filter_binding_sites(
    mol_list1: List[Tuple[str, "Molecule"]],
    mol_list2: List[Tuple[str, "Molecule"]],
    cut_off_val: float,
    rmsd_symm: bool,
    logger: Optional[logging.Logger],
    ligand_size: int,
    grid: float = 0.5,
    nb_radius: int = 3,
    dm_tau: float = 1e-3,
    atoms_map: bool = True,
    use_fallback_allrefs: bool = True,
) -> List[Tuple[str, "Molecule"]]:
    """
    Duplicate filter with OR gating:
      - Proceed to RMSD if EITHER voxel-neighbor OR ligand pair-distance gate passes.
      - Neighbor search uses ligand COM voxels (grid A, Chebyshev radius nb_radius).
      - Pair-distance gate is permutation- & rigid-motion-invariant (tolerance dm_tau).
      - Conditional mirror RMSD (mirror only if needed).
      - Fallback: if no neighbor match, scan all refs using the pair-distance gate.

    Returns list of (name, Molecule) from mol_list1 that are unique.
    """
    info = (logger.info if logger else print)

    info("\nStarting filtering procedure:")
    info("Comparing within structures found in this round:")

    # Precompute keys for new list
    l1 = [(name, mol, _key_for_grid(mol, ligand_size, grid)) for (name, mol) in mol_list1]

    # --- 1) Within-round dedup with OR gating ---
    to_remove_tmp = set()
    for i in range(len(l1)):
        ni, mi, ki = l1[i]
        if ni in to_remove_tmp:
            continue
        for j in range(i + 1, len(l1)):
            nj, mj, kj = l1[j]
            if nj in to_remove_tmp:
                continue

            voxel_ok = _cheb_within(ki, kj, nb_radius)
            dm_ok = ligand_pairdist_ok(mi, mj, ligand_size, tau=dm_tau)

            if not (voxel_ok or dm_ok):
                continue  # neither gate passes -> skip

            r, rm = compute_rmsd_conditional(mi, mj, rmsd_symm, cut_off_val, atoms_map=atoms_map)
            if min(r, rm) < cut_off_val:
                info(f"Duplicate found: {ni} vs {nj}, RMSD: {min(r, rm):.3f}")
                to_remove_tmp.add(nj)

    unique_tmp = [(name, mol) for (name, mol, _) in l1 if name not in to_remove_tmp]

    # --- 2) Against reference set (neighbors + OR + fallback) ---
    info("Comparing with structures already present in the Optimization Dataset")

    # Bucket refs
    buckets2 = defaultdict(list)
    for rname, rmol in mol_list2:
        buckets2[_key_for_grid(rmol, ligand_size, grid)].append((rname, rmol))

    to_remove_final = set()
    for name, mol in unique_tmp:
        key = _key_for_grid(mol, ligand_size, grid)

        # Neighbor candidates
        candidates = []
        for nk in _neighbor_keys(key, nb_radius):
            if nk in buckets2:
                candidates.extend(buckets2[nk])

        found = False
        # Neighbor pass (OR gating vs each candidate)
        for rname, rmol in candidates:
            voxel_ok = True  # by construction: neighbor list
            dm_ok = ligand_pairdist_ok(mol, rmol, ligand_size, tau=dm_tau)
            if not (voxel_ok or dm_ok):
                continue
            r, rm = compute_rmsd_conditional(mol, rmol, rmsd_symm, cut_off_val, atoms_map=atoms_map)
            if min(r, rm) < cut_off_val:
                info(f"Duplicate found: {name} vs. {rname}, RMSD: {min(r, rm):.3f}")
                to_remove_final.add(name)
                found = True
                break

        # Fallback: scan ALL refs, but still prefilter by pair-distance (fast)
        if use_fallback_allrefs and not found:
            for rname, rmol in mol_list2:
                dm_ok = ligand_pairdist_ok(mol, rmol, ligand_size, tau=dm_tau)
                if not dm_ok:
                    continue
                r, rm = compute_rmsd_conditional(mol, rmol, rmsd_symm, cut_off_val, atoms_map=atoms_map)
                if min(r, rm) < cut_off_val:
                    info(f"Duplicate found (fallback): {name} vs. {rname}, RMSD: {min(r, rm):.3f}")
                    to_remove_final.add(name)
                    found = True
                    break

    total_removed = len(to_remove_tmp) + len(to_remove_final)
    unique_final = [pair for pair in unique_tmp if pair[0] not in to_remove_final]
    info(f"{total_removed} duplicates removed. {len(unique_final)} unique binding sites remain "
         f"(grid={grid}, nb_radius={nb_radius}, dm_tau={dm_tau}).")
    return unique_final
