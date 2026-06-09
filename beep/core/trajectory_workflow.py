"""Trajectory-benchmark orchestration helpers.

Workflow-level glue for the ``geom_benchmark`` trajectory analysis: for
each DFT functional, evaluate SP + gradient at every geometry along the
reference optimization trajectory and report MAE/RMSE of energy
(meV/atom) and forces (meV/Å) against the reference, plus a combined
z-score-weighted ranking.

The pure-math layer lives in :mod:`beep.core.trajectory_metrics`
(``per_step_deltas``, ``summarize_method_metrics``,
``combined_zscore_ranking``); the QCFractal I/O lives in
:mod:`beep.adapters.qcfractal_adapter`. This module orchestrates the
two against a benchmark's reference + DFT optimizations.

Inspired by Bovolenta et al. 2025 (arXiv 2508.14219), Appendix C.2 /
Fig C.1.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import is_complete, is_incomplete, is_error
from .logging_utils import (
    padded_log,
    log_trajectory_metrics_per_group,
    log_trajectory_ranking_table,
)
from .plotting_utils import trajectory_error_histograms
from .trajectory_metrics import (
    per_step_deltas, summarize_method_metrics, combined_zscore_ranking,
)

bcheck = "✔"


def collect_reference_trajectories(
    odset_dict: dict, geom_ref_opt_lot: str, logger: logging.Logger,
) -> dict:
    """Pull each system's reference optimization trajectory.

    A fetch failure on one system (e.g. qcportal cache OverflowError on
    a record with corrupted property values) is logged with a prominent
    warning and the system is skipped — other systems still get analysed.
    Matches the defense-in-depth pattern of skipping a single bad record
    rather than aborting the whole workflow.
    """
    out = {}
    for struct_name, odset in odset_dict.items():
        try:
            steps = qcf.get_optimization_trajectory(
                odset, struct_name, geom_ref_opt_lot,
            )
        except Exception as exc:
            logger.warning("")
            logger.warning(
                f"  !! WARNING: {struct_name}: trajectory fetch failed "
                f"({type(exc).__name__}: {exc})."
            )
            logger.warning(
                f"  !!          This system will be EXCLUDED from the "
                f"trajectory benchmark; eq-geometry RMSD is unaffected."
            )
            logger.warning("")
            continue
        if not steps:
            logger.warning(
                f"  No reference trajectory for {struct_name} at "
                f"{geom_ref_opt_lot} — skipping system."
            )
            continue
        if any(s["gradient_hartree_per_bohr"] is None for s in steps):
            logger.warning(
                f"  Reference trajectory for {struct_name} missing gradient "
                f"on some steps — skipping system."
            )
            continue
        out[struct_name] = steps
        logger.info(
            f"  {struct_name}: {len(steps)} reference trajectory steps."
        )
    return out


def build_trajectory_sp_datasets(
    client, traj_data: dict, all_dft_functionals: List[str],
    program: str, dft_keyword, logger: logging.Logger,
) -> dict:
    """Per system: create ``<system>_trajref`` SinglepointDataset, add one
    entry per traj step (``step_NNN``), and register one SP+gradient spec
    per DFT functional. Idempotent — safe to re-run."""
    traj_dsets = {}
    kw = dft_keyword if isinstance(dft_keyword, dict) else {}
    for struct_name, steps in traj_data.items():
        ds_name = f"{struct_name}_trajref"
        ds_sp = qcf.get_or_create_singlepoint_dataset(client, ds_name)

        entries = [
            (f"step_{i:03d}", step["molecule"])
            for i, step in enumerate(steps)
        ]
        qcf.add_singlepoint_entries(ds_sp, entries)

        for functional in all_dft_functionals:
            method, basis = functional.split("_", 1)
            qcf.add_gradient_spec(
                ds_sp,
                spec_name=functional,
                method=method,
                basis=basis,
                program=program,
                keywords=kw,
                description=(
                    f"SP+gradient on ref trajectory geometry "
                    f"for {functional}"
                ),
            )
        traj_dsets[struct_name] = ds_sp
        logger.info(
            f"  {struct_name}: dataset '{ds_name}' "
            f"({len(entries)} entries, {len(all_dft_functionals)} specs)"
        )
    return traj_dsets


def submit_trajectory_singlepoints(
    traj_dsets: dict, all_dft_functionals: List[str],
    tag: str, logger: logging.Logger,
):
    """Submit all (entry × spec) SP+gradient pairs.

    Assumes ``all_dft_functionals`` is already lowercased (per the
    convention applied at the top of ``run_trajectory_analysis``,
    matching be_hess / energy_benchmark / sampling).
    """
    inserted = existing = 0
    for ds_sp in traj_dsets.values():
        meta = qcf.submit_singlepoints_in_dataset(
            ds_sp, spec_names=all_dft_functionals, tag=tag,
        )
        inserted += getattr(meta, "n_inserted", 0)
        existing += getattr(meta, "n_existing", 0)
    logger.info(
        f"  Submitted {inserted} new SP+gradient computations "
        f"({existing} already existed) to tag '{tag}'."
    )
    return inserted, existing


def wait_for_trajectory_completion(
    traj_dsets: dict, all_dft_functionals: List[str],
    wait_interval: int, logger: logging.Logger,
):
    """Poll until every (system × entry × spec) trajectory SP is terminal.

    Assumes ``all_dft_functionals`` is already lowercased.
    """
    while True:
        complete = incomplete = error = 0
        for ds_sp in traj_dsets.values():
            for entry_name in ds_sp.entry_names:
                for spec_key in all_dft_functionals:
                    record = ds_sp.get_record(entry_name, spec_key)
                    if record is None:
                        continue
                    if is_complete(record.status):
                        complete += 1
                    elif is_incomplete(record.status):
                        incomplete += 1
                    elif is_error(record.status):
                        error += 1
        if incomplete == 0:
            logger.info(
                f"  Traj SP: Complete: {complete}, Error: {error} {bcheck}"
            )
            return complete, error
        logger.info(
            f"  Traj SP: {complete} done, {incomplete} running, {error} err"
        )
        logger.info(f"  Waiting {wait_interval}s before rechecking...")
        time.sleep(wait_interval)


def compute_per_method_trajectory_metrics(
    traj_data: dict, traj_dsets: dict, all_dft_functionals: List[str],
    logger: logging.Logger,
):
    """For each functional, aggregate per-step deltas across systems and
    return ``(metrics, raw_deltas)``.

    ``metrics`` is a dict ``{functional → summarised MAE/RMSE numbers}``;
    ``raw_deltas`` is a dict ``{functional → {delta_e arrays, delta_force
    arrays}}`` for downstream plotting. Skips functionals with no
    complete records.
    """
    metrics: Dict[str, dict] = {}
    raw_deltas: Dict[str, dict] = {}

    for functional in all_dft_functionals:
        per_system = []
        skipped_systems = 0

        for struct_name, steps in traj_data.items():
            ds_sp = traj_dsets.get(struct_name)
            if ds_sp is None:
                continue

            ref_e, ref_g, dft_e, dft_g = [], [], [], []
            n_atoms = None
            for i, step in enumerate(steps):
                entry_name = f"step_{i:03d}"
                de, dg = qcf.fetch_sp_energy_gradient(
                    ds_sp, entry_name, functional,
                )
                if de is None or dg is None:
                    continue
                if (step["energy_hartree"] is None
                        or step["gradient_hartree_per_bohr"] is None):
                    continue
                if n_atoms is None:
                    n_atoms = len(step["molecule"].symbols)
                ref_e.append(step["energy_hartree"])
                ref_g.append(step["gradient_hartree_per_bohr"])
                dft_e.append(de)
                dft_g.append(dg)

            if not ref_e:
                skipped_systems += 1
                continue

            per_system.append(per_step_deltas(
                np.asarray(ref_e), np.asarray(ref_g),
                np.asarray(dft_e), np.asarray(dft_g),
                n_atoms=n_atoms,
            ))

        if not per_system:
            logger.warning(
                f"  {functional}: no complete trajectory data — excluded."
            )
            continue

        metrics[functional] = summarize_method_metrics(per_system)
        raw_deltas[functional] = {
            "delta_e_per_atom_meV": np.concatenate(
                [d["delta_e_per_atom_meV"] for d in per_system]
            ),
            "delta_force_meV_per_A": np.concatenate(
                [d["delta_force_meV_per_A"].ravel() for d in per_system]
            ),
        }
        if skipped_systems:
            logger.info(
                f"  {functional}: {skipped_systems} systems had no complete "
                "records."
            )
    return metrics, raw_deltas


def build_ranking_df(
    rmsd_df: pd.DataFrame, traj_metrics: dict,
    dft_geom_functionals: dict, score_weights: dict,
):
    """Merge eq-RMSD and trajectory force-RMSD keyed by functional and
    compute the z-score combined ranking.

    ``rmsd_df`` columns are mixed-case (built by ``compare_all_rmsd``);
    ``traj_metrics`` keys are lowercase (the workflow convention). We
    look up in mixed case but emit lowercase keys so the intersection
    with ``traj_metrics`` is well-defined.

    Energy is not part of the geom_benchmark ranking — absolute energies
    aren't meaningful for geometry quality. ``rmsd_energy`` stays
    available in ``traj_metrics`` as a side-channel diagnostic.
    """
    rmsd_eq = {}
    for group, funcs in dft_geom_functionals.items():
        for f in funcs:
            col = f"{group}_{f}"
            if col in rmsd_df.columns:
                mean_val = float(rmsd_df[col].dropna().mean())
                if not np.isnan(mean_val):
                    rmsd_eq[f.lower()] = mean_val

    common = sorted(set(rmsd_eq.keys()) & set(traj_metrics.keys()))
    if not common:
        return None

    metrics_df = pd.DataFrame(
        [
            {
                "rmsd_eq": rmsd_eq[m],
                "rmsd_force": traj_metrics[m]["rmsd_force"],
            }
            for m in common
        ],
        index=common,
    )
    ranking_df = combined_zscore_ranking(metrics_df, score_weights)
    return ranking_df.join(metrics_df, how="left")


def run_trajectory_analysis(
    *, config, client, odset_dict: dict, geom_ref_opt_lot: str,
    all_dft_functionals: List[str], dft_geom_functionals: dict,
    dft_program: str, dft_keyword, dft_tag: str,
    rmsd_df: pd.DataFrame, res_folder: Path, logger: logging.Logger,
):
    """Top-level driver for the trajectory benchmark.

    Pulls each system's reference trajectory, builds a per-system
    ``SinglepointDataset``, registers one SP+gradient spec per DFT
    functional, submits, waits for completion, computes per-method
    MAE/RMSE of energy and forces vs the reference, builds the combined
    z-score-weighted ranking, logs the BENCHMARK-RESULTS-style per-group
    tables + the ranking table, and writes the JSON + plot artifacts
    next to the existing geom_benchmark outputs.

    Returns ``(ranking_df, raw_deltas)``.
    """
    # Convention used by be_hess / energy_benchmark / sampling: lowercase
    # spec names once at the workflow entry, then pass through untouched.
    # qcf-astrochem stores specs lowercase and lookups are case-sensitive.
    # dft_geom_functionals stays mixed-case here because compare_all_rmsd
    # built rmsd_df with mixed-case column names; build_ranking_df does
    # the lookup in mixed case and emits lowercase keys.
    all_dft_functionals = [s.lower() for s in all_dft_functionals]
    padded_log(logger, "Trajectory analysis (DFT vs reference)")

    logger.info("\nCollecting reference trajectories…")
    traj_data = collect_reference_trajectories(
        odset_dict, geom_ref_opt_lot, logger,
    )
    if not traj_data:
        logger.warning(
            "\n  No usable reference trajectories — skipping trajectory "
            "analysis.\n"
        )
        return None, {}

    logger.info("\nBuilding trajectory singlepoint datasets…")
    traj_dsets = build_trajectory_sp_datasets(
        client, traj_data, all_dft_functionals,
        dft_program, dft_keyword, logger,
    )

    logger.info(
        f"\nSubmitting trajectory SP+gradient computations to tag "
        f"'{dft_tag}'…"
    )
    submit_trajectory_singlepoints(
        traj_dsets, all_dft_functionals, dft_tag, logger,
    )

    logger.info("\nWaiting for trajectory SPs to complete…")
    wait_for_trajectory_completion(
        traj_dsets, all_dft_functionals, wait_interval=200, logger=logger,
    )

    logger.info("\nComputing per-method trajectory metrics…")
    traj_metrics, raw_deltas = compute_per_method_trajectory_metrics(
        traj_data, traj_dsets, all_dft_functionals, logger,
    )

    ranking_df = build_ranking_df(
        rmsd_df, traj_metrics, dft_geom_functionals, config.score_weights,
    )

    padded_log(logger, "TRAJECTORY BENCHMARK RESULTS")
    log_trajectory_metrics_per_group(
        logger, traj_metrics, dft_geom_functionals, ranking_df,
    )

    padded_log(logger, "Combined ranking (z-score weighted)")
    log_trajectory_ranking_table(
        logger, ranking_df, score_weights=config.score_weights,
    )

    folder_path_json = res_folder / "json_data"
    folder_path_json.mkdir(parents=True, exist_ok=True)
    metrics_json = {
        m: {k: v for k, v in d.items()}
        for m, d in traj_metrics.items()
    }
    (folder_path_json / "results_trajectory_metrics.json").write_text(
        json.dumps(metrics_json, indent=4, default=str)
    )
    if ranking_df is not None:
        ranking_df.to_json(
            str(folder_path_json / "results_trajectory_ranking.json"),
        )

    # Persist raw per-functional force/energy delta arrays for downstream
    # histogram / distribution analysis. Each functional contributes two
    # flat arrays — `<functional>__force` (meV/Å, per Cartesian component)
    # and `<functional>__energy` (meV/atom). Load with
    # ``data = np.load(path)``; iterate with ``data.files``.
    if raw_deltas:
        npz_payload = {}
        for functional, d in raw_deltas.items():
            npz_payload[f"{functional}__force"] = np.asarray(
                d["delta_force_meV_per_A"]
            )
            npz_payload[f"{functional}__energy"] = np.asarray(
                d["delta_e_per_atom_meV"]
            )
        np.savez_compressed(
            folder_path_json / "raw_deltas_trajectory.npz", **npz_payload,
        )

    logger.info(
        f"\n  Trajectory metrics + ranking + raw deltas written to "
        f"{folder_path_json}/\n"
    )

    if raw_deltas:
        folder_path_plots = res_folder / "plots"
        folder_path_plots.mkdir(parents=True, exist_ok=True)
        try:
            trajectory_error_histograms(
                raw_deltas, mol_name=config.molecule,
                plot_path=str(folder_path_plots),
                ranking_df=ranking_df,
            )
            logger.info(
                f"  Trajectory error histograms written to "
                f"{folder_path_plots}/\n"
            )
        except Exception as exc:
            logger.warning(
                f"  Histogram plotting failed: {exc} — "
                f"metrics are still available in json_data/"
            )

    return ranking_df, raw_deltas
