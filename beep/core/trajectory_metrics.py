"""
Trajectory benchmark metrics — pure computation, no QCFractal deps.

Energies are reported in meV/atom; gradient components (forces) in meV/Å.
This matches the units used in MLP-validation literature (Bovolenta
et al. 2025, A&A, Fig C.1) and makes the metrics directly comparable
across cluster sizes.

Conversion factors are taken from CODATA 2018 / qcelemental.
"""
from typing import Dict, Iterable

import numpy as np
import pandas as pd


HARTREE_TO_MEV = 27211.386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_MEV_PER_A = HARTREE_TO_MEV / BOHR_TO_ANGSTROM


def per_step_deltas(
    ref_energies,
    ref_gradients,
    dft_energies,
    dft_gradients,
    n_atoms: int,
) -> Dict[str, np.ndarray]:
    """Compute per-trajectory-step (DFT − reference) deltas in MLP units.

    Parameters
    ----------
    ref_energies, dft_energies : array-like, shape (n_steps,)
        Total energies in hartree at each trajectory step.
    ref_gradients, dft_gradients : array-like, shape (n_steps, n_atoms, 3)
        Cartesian gradients in hartree/bohr at each trajectory step.
    n_atoms : int
        Number of atoms in the supermolecule, for per-atom energy
        normalization.

    Returns
    -------
    dict with keys:
      ``delta_e_per_atom_meV``  — shape (n_steps,), meV/atom
      ``delta_force_meV_per_A`` — shape (n_steps, n_atoms, 3), meV/Å
    """
    ref_e = np.asarray(ref_energies, dtype=float)
    dft_e = np.asarray(dft_energies, dtype=float)
    ref_g = np.asarray(ref_gradients, dtype=float)
    dft_g = np.asarray(dft_gradients, dtype=float)

    if ref_e.shape != dft_e.shape:
        raise ValueError(
            f"ref_energies shape {ref_e.shape} != dft_energies shape {dft_e.shape}"
        )
    if ref_g.shape != dft_g.shape:
        raise ValueError(
            f"ref_gradients shape {ref_g.shape} != dft_gradients shape {dft_g.shape}"
        )
    if n_atoms <= 0:
        raise ValueError(f"n_atoms must be > 0, got {n_atoms}")

    delta_e_per_atom_meV = (dft_e - ref_e) * HARTREE_TO_MEV / n_atoms
    delta_force_meV_per_A = (dft_g - ref_g) * HARTREE_PER_BOHR_TO_MEV_PER_A
    return {
        "delta_e_per_atom_meV": delta_e_per_atom_meV,
        "delta_force_meV_per_A": delta_force_meV_per_A,
    }


def summarize_method_metrics(
    per_system_deltas: Iterable[Dict[str, np.ndarray]]
) -> Dict[str, float]:
    """Aggregate per-system delta arrays for one functional and summarize.

    The energy and force deltas are concatenated across systems
    (a trajectory of length N_i contributes N_i energy points and
    3 · N_i · n_atoms_i force-component points), then reduced to
    root-mean-square deviation (RMSD).

    RMSD vs the reference is the right metric here: same mathematical
    form as the equilibrium-geometry ``compare_rmsd``; outlier-sensitive
    (a single bad gradient point — the failure mode that matters for
    geometry optimization — surfaces clearly).

    Force RMSD is taken over every Cartesian component (matching the
    Bovolenta 2025 Fig C.1 convention). Energy RMSD is computed for
    completeness — the geom_benchmark workflow does not use it (absolute
    energies aren't meaningful for geometry ranking; use the
    energy_benchmark workflow for relative-energy comparison).
    """
    per_system_deltas = list(per_system_deltas)
    if not per_system_deltas:
        return {
            "rmsd_energy": float("nan"),
            "rmsd_force": float("nan"),
            "n_energy_points": 0,
            "n_force_components": 0,
        }

    all_de = np.concatenate(
        [np.asarray(d["delta_e_per_atom_meV"]).ravel() for d in per_system_deltas]
    )
    all_df = np.concatenate(
        [np.asarray(d["delta_force_meV_per_A"]).ravel() for d in per_system_deltas]
    )

    return {
        "rmsd_energy": float(np.sqrt(np.mean(all_de ** 2))),
        "rmsd_force": float(np.sqrt(np.mean(all_df ** 2))),
        "n_energy_points": int(all_de.size),
        "n_force_components": int(all_df.size),
    }


def combined_zscore_ranking(
    metrics_df: pd.DataFrame, weights: Dict[str, float]
) -> pd.DataFrame:
    """Combine per-method metrics into a weighted z-score ranking.

    The metrics combined are taken from ``weights.keys()`` — the same
    function therefore serves the 2-metric geom-benchmark case
    (``rmsd_eq`` + ``rmsd_force``) and any future N-metric variant
    (e.g. energy_benchmark) without modification.

    For each metric column, computes ``z = (x − mean) / std`` across the
    methods present in ``metrics_df`` (population std, ``ddof=0``).
    Lower combined score = better functional.

    Parameters
    ----------
    metrics_df : DataFrame
        Index = method name. Columns must include every key of
        ``weights``.
    weights : dict
        ``{metric_name → float}``. Determines which columns are combined
        and how heavily each is weighted.

    Returns
    -------
    DataFrame indexed by method, with z-score columns (named
    ``z_<metric>``) plus ``combined_score``, sorted ascending by
    ``combined_score``.

    Notes
    -----
    A metric column where all methods are equal (zero std) contributes
    ``0`` to every z-score — that dimension carries no ranking signal.
    """
    if not weights:
        raise ValueError("weights must be non-empty")
    metric_cols = list(weights.keys())
    for col in metric_cols:
        if col not in metrics_df.columns:
            raise ValueError(f"metrics_df missing required column: {col}")

    out = pd.DataFrame(index=metrics_df.index)
    for col in metric_cols:
        x = metrics_df[col].astype(float)
        mean = float(x.mean())
        std = float(x.std(ddof=0))
        if std == 0 or np.isnan(std):
            out[f"z_{col}"] = 0.0
        else:
            out[f"z_{col}"] = (x - mean) / std

    out["combined_score"] = sum(
        float(weights[col]) * out[f"z_{col}"] for col in metric_cols
    )
    return out.sort_values("combined_score")
