"""Normal-mode sampling benchmark orchestration helpers.

Workflow-level glue for the ``nm_sampling`` workflow: per binding site,
compute a Hessian → diagonalise → classify normal modes via fragment-COM
projection → pick lowest-frequency modes per band → generate ± displaced
geometries → submit CCSD(T) gradients (reference) plus DFT gradients per
functional on every displacement → report per-functional force-RMSD vs
CCSD(T), per-category.

The pure-math layer lives in :mod:`beep.core.normal_mode_sampling`
(classify/select/displace + the qcelemental harmonic-analysis wrapper);
the QCFractal I/O lives in :mod:`beep.adapters.qcfractal_adapter`. This
module chains them.

The output format mirrors the trajectory-benchmark blueprint
(:mod:`beep.core.trajectory_workflow`): per-system ``SinglepointDataset``
named ``<system>_nmsamp``, entries ``disp_NNN`` per displaced structure,
one gradient spec per functional plus a reference-CCSD(T) spec. The
per-category MAE / force-RMSD reporting reuses
``log_trajectory_metrics_per_group``.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from qcelemental.models import Molecule
except ImportError:  # pragma: no cover
    from qcelemental.models.molecule import Molecule

from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import is_complete, is_incomplete, is_error
from .logging_utils import padded_log, log_trajectory_metrics_per_group
from .normal_mode_sampling import (
    classify_mode, select_modes, displace_along_mode,
    write_molden, write_modes_json,
)
from .trajectory_metrics import per_step_deltas, summarize_method_metrics

bcheck = "✔"
gear = "⚙"

# Reference spec key. Set to the lowercased reference_grad_lot at run time;
# the orchestrator stitches the value through so the per-method fetch can
# pair the DFT result with the matching reference.
_REF_SPEC_NAME = None


# ---------------------------------------------------------------------------
# Step 1 — collect optimized geometries
# ---------------------------------------------------------------------------

def collect_optimized_geometries(
    odset_dict: dict, geometry_opt_lot: str, logger: logging.Logger,
) -> Dict[str, Molecule]:
    """Pull each system's binding-site optimised geometry as a Molecule."""
    out: Dict[str, Molecule] = {}
    for struct_name, odset in odset_dict.items():
        try:
            record = odset.get_record(struct_name, geometry_opt_lot.lower())
        except Exception as exc:
            logger.warning(
                f"  !! WARNING: {struct_name}: failed to fetch optimised "
                f"geometry record ({type(exc).__name__}: {exc}) — skipping."
            )
            continue
        if record is None or not is_complete(record.status):
            logger.warning(
                f"  {struct_name}: no complete optimisation at "
                f"{geometry_opt_lot} — skipping."
            )
            continue
        out[struct_name] = record.final_molecule
        logger.info(
            f"  {struct_name}: equilibrium geometry fetched "
            f"({len(record.final_molecule.symbols)} atoms)."
        )
    return out


# ---------------------------------------------------------------------------
# Step 2 — submit Hessians
# ---------------------------------------------------------------------------

def submit_system_hessians(
    client, ref_mols: Dict[str, Molecule],
    hessian_lot: str, hessian_program: str,
    hessian_keywords, tag_hessian: str,
    logger: logging.Logger,
) -> List[int]:
    """Submit a Hessian SP for each system's equilibrium molecule."""
    lot_parts = hessian_lot.split("_", 1)
    method = lot_parts[0]
    basis = lot_parts[1] if len(lot_parts) == 2 else None

    mol_ids = [m.id for m in ref_mols.values() if m.id is not None]
    if not mol_ids:
        logger.warning("  No molecule IDs to submit Hessians for.")
        return []

    meta, record_ids = qcf.submit_hessians(
        client=client, program=hessian_program,
        method=method, basis=basis,
        mol_ids=mol_ids, keywords=hessian_keywords or {},
        tag=tag_hessian,
    )
    logger.info(
        f"  Submitted {meta.n_inserted} new Hessian(s) at {method}/{basis} "
        f"({meta.n_existing} already existed) to tag '{tag_hessian}'."
    )
    return record_ids


# ---------------------------------------------------------------------------
# Step 3 — fetch + classify normal modes
# ---------------------------------------------------------------------------

def collect_system_normal_modes(
    client, ref_mols: Dict[str, Molecule], hessian_lot: str,
    n_adsorbate_atoms: int, inter_threshold: float, bend_max_cm: float,
    logger: logging.Logger,
) -> Dict[str, dict]:
    """Per system, fetch the Hessian record and classify each normal mode.

    Returns ``{struct_name → {frequencies_cm, modes_cart, classes, mol}}``.
    Systems with missing Hessians are skipped.
    """
    out: Dict[str, dict] = {}
    for struct_name, mol in ref_mols.items():
        if mol.id is None:
            logger.warning(
                f"  {struct_name}: molecule has no server ID — skipping "
                "normal-mode fetch."
            )
            continue
        try:
            freqs_cm, modes_cart, mol_from_record = qcf.fetch_normal_modes(
                client, mol.id, hessian_lot,
            )
        except Exception as exc:
            logger.warning(
                f"  !! WARNING: {struct_name}: failed to fetch normal modes "
                f"({type(exc).__name__}: {exc}) — skipping."
            )
            continue
        if freqs_cm is None or modes_cart is None or len(freqs_cm) == 0:
            logger.warning(
                f"  {struct_name}: no normal-mode data — skipping."
            )
            continue

        masses = np.asarray(mol_from_record.masses)
        positions = np.asarray(mol_from_record.geometry).reshape(-1, 3)
        classes = []
        for i, freq in enumerate(freqs_cm):
            cls = classify_mode(
                mode_cart=modes_cart[i],
                masses=masses,
                positions=positions,
                n_adsorbate_atoms=n_adsorbate_atoms,
                frequency_cm=float(np.real(freq)),
                inter_threshold=inter_threshold,
                bend_max_cm=bend_max_cm,
            )
            classes.append(cls)

        # Count classes for the log line
        counts = {b: classes.count(b) for b in ("intermolecular", "bending", "stretching")}
        logger.info(
            f"  {struct_name}: {len(freqs_cm)} vib modes — "
            f"intermolecular={counts['intermolecular']}, "
            f"bending={counts['bending']}, "
            f"stretching={counts['stretching']}"
        )

        out[struct_name] = {
            "frequencies_cm": freqs_cm,
            "modes_cart": modes_cart,
            "classes": classes,
            "mol": mol_from_record,
        }
    return out


# ---------------------------------------------------------------------------
# Step 4 — select displacements + build displaced molecules
# ---------------------------------------------------------------------------

def build_displaced_molecules(
    mode_data: Dict[str, dict], bands: Dict[str, "dict"],
    freq_max_imag_cm: float,
    extra_amplitudes_lowest_count: int, extra_amplitude_factor: float,
    logger: logging.Logger,
) -> Dict[str, List[Tuple[str, Molecule, dict]]]:
    """Per system, apply the per-band caps and generate ± displaced Molecules.

    Returns ``{struct_name → [(entry_name, Molecule, metadata), ...]}`` where
    ``metadata`` records the (mode_index, amplitude_A, sign, band, frequency_cm)
    of each displacement for downstream logging.
    """
    band_caps = {b: spec.cap for b, spec in bands.items()}
    band_amps = {b: spec.amplitude_A for b, spec in bands.items()}

    out: Dict[str, List[Tuple[str, Molecule, dict]]] = {}
    for struct_name, data in mode_data.items():
        freqs = data["frequencies_cm"]
        modes = data["modes_cart"]
        classes = data["classes"]
        mol = data["mol"]

        picks = select_modes(
            frequencies_cm=freqs, classes=classes,
            band_caps=band_caps, band_amplitudes=band_amps,
            freq_max_imag_cm=freq_max_imag_cm,
            extra_amplitudes_lowest_count=extra_amplitudes_lowest_count,
            extra_amplitude_factor=extra_amplitude_factor,
        )
        if not picks:
            logger.warning(
                f"  {struct_name}: no modes survived selection — skipping."
            )
            continue

        geom_bohr = np.asarray(mol.geometry).reshape(-1, 3)
        entries: List[Tuple[str, Molecule, dict]] = []
        counter = 0
        for mode_idx, amp_A, band in picks:
            for sign in (+1, -1):
                new_geom = displace_along_mode(
                    geometry_bohr=geom_bohr,
                    mode_cart=modes[mode_idx],
                    amplitude_A=amp_A,
                    sign=sign,
                )
                disp_mol = Molecule(
                    symbols=mol.symbols,
                    geometry=new_geom.flatten(),
                    masses=mol.masses,
                    molecular_charge=mol.molecular_charge,
                    molecular_multiplicity=mol.molecular_multiplicity,
                    fix_com=True, fix_orientation=True,
                )
                entry_name = f"disp_{counter:03d}"
                entries.append((
                    entry_name, disp_mol,
                    {
                        "mode_idx": int(mode_idx),
                        "amplitude_A": float(amp_A),
                        "sign": int(sign),
                        "band": band,
                        "frequency_cm": float(np.real(freqs[mode_idx])),
                    },
                ))
                counter += 1
        logger.info(
            f"  {struct_name}: {len(picks)} mode picks → "
            f"{len(entries)} displaced structures"
        )
        out[struct_name] = entries
    return out


# ---------------------------------------------------------------------------
# Step 5 — build per-system SinglepointDatasets
# ---------------------------------------------------------------------------

def build_nm_sp_datasets(
    client, displaced: Dict[str, List[Tuple[str, Molecule, dict]]],
    all_dft_functionals: List[str], reference_spec_name: str,
    reference_grad_lot: str, reference_grad_program: str,
    reference_grad_keywords, dft_program: str, dft_keyword,
    logger: logging.Logger,
):
    """Per system, create the `<system>_nmsamp` SinglepointDataset, add
    entries, register one gradient spec per DFT functional + one reference
    CCSD(T) spec. Idempotent."""
    sp_dsets: Dict[str, object] = {}
    dft_kw = dft_keyword if isinstance(dft_keyword, dict) else {}
    ref_kw = reference_grad_keywords if isinstance(reference_grad_keywords, dict) else {}

    ref_method, _, ref_basis = reference_grad_lot.partition("_")
    if not ref_basis:
        ref_basis = None

    for struct_name, entries in displaced.items():
        ds_name = f"{struct_name}_nmsamp"
        ds_sp = qcf.get_or_create_singlepoint_dataset(client, ds_name)

        qcf.add_singlepoint_entries(
            ds_sp, [(name, mol) for name, mol, _meta in entries],
        )

        # DFT specs
        for functional in all_dft_functionals:
            method, basis = functional.split("_", 1)
            qcf.add_gradient_spec(
                ds_sp, spec_name=functional, method=method, basis=basis,
                program=dft_program, keywords=dft_kw,
                description=(
                    f"NM displacement gradient at {functional} for "
                    f"{struct_name}"
                ),
            )

        # Reference CCSD(T) spec
        qcf.add_gradient_spec(
            ds_sp, spec_name=reference_spec_name,
            method=ref_method, basis=ref_basis,
            program=reference_grad_program, keywords=ref_kw,
            description=f"NM displacement reference gradient at {reference_grad_lot}",
        )

        sp_dsets[struct_name] = ds_sp
        logger.info(
            f"  {struct_name}: dataset '{ds_name}' "
            f"({len(entries)} entries, {len(all_dft_functionals)} DFT specs "
            f"+ 1 reference spec)"
        )
    return sp_dsets


# ---------------------------------------------------------------------------
# Step 6 — submit gradient SPs
# ---------------------------------------------------------------------------

def submit_nm_singlepoints(
    sp_dsets: dict, all_dft_functionals: List[str], reference_spec_name: str,
    tag_dft_grad: str, tag_reference_grad: str,
    logger: logging.Logger,
):
    """Submit DFT specs to the DFT tag and the reference spec to the
    reference tag. Returns counts for logging."""
    inserted_dft = existing_dft = 0
    inserted_ref = existing_ref = 0
    for ds_sp in sp_dsets.values():
        meta_dft = qcf.submit_singlepoints_in_dataset(
            ds_sp, spec_names=all_dft_functionals, tag=tag_dft_grad,
        )
        inserted_dft += getattr(meta_dft, "n_inserted", 0)
        existing_dft += getattr(meta_dft, "n_existing", 0)

        meta_ref = qcf.submit_singlepoints_in_dataset(
            ds_sp, spec_names=[reference_spec_name], tag=tag_reference_grad,
        )
        inserted_ref += getattr(meta_ref, "n_inserted", 0)
        existing_ref += getattr(meta_ref, "n_existing", 0)
    logger.info(
        f"  DFT gradients: {inserted_dft} new, {existing_dft} existing (tag '{tag_dft_grad}')."
    )
    logger.info(
        f"  Ref gradients: {inserted_ref} new, {existing_ref} existing (tag '{tag_reference_grad}')."
    )


# ---------------------------------------------------------------------------
# Step 7 — wait for completion
# ---------------------------------------------------------------------------

def wait_for_nm_completion(
    sp_dsets: dict, all_spec_names: List[str], wait_interval: int,
    logger: logging.Logger,
):
    """Poll until every (system × entry × spec) record is terminal.

    ``all_spec_names`` should include both the DFT functionals and the
    reference spec name (all lowercase).
    """
    while True:
        complete = incomplete = error = 0
        for ds_sp in sp_dsets.values():
            for entry_name in ds_sp.entry_names:
                for spec_key in all_spec_names:
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
                f"  NM SP: Complete: {complete}, Error: {error} {bcheck}"
            )
            return complete, error
        logger.info(
            f"  NM SP: {complete} done, {incomplete} running, {error} err"
        )
        logger.info(f"  Waiting {wait_interval}s before rechecking...")
        time.sleep(wait_interval)


# ---------------------------------------------------------------------------
# Step 8 — compute per-method force-RMSD vs reference
# ---------------------------------------------------------------------------

def compute_per_method_nm_metrics(
    displaced: Dict[str, List[Tuple[str, Molecule, dict]]],
    sp_dsets: dict, all_dft_functionals: List[str],
    reference_spec_name: str, logger: logging.Logger,
):
    """For each DFT functional, compare its gradient to the reference at
    every displaced geometry across every system. Returns
    ``(metrics_dict, raw_deltas_dict)`` matching the trajectory layout.

    Only the force component is meaningful here (energies at displaced
    geometries aren't a benchmark target on their own). We reuse
    ``per_step_deltas`` + ``summarize_method_metrics`` from
    ``trajectory_metrics``; the energy entries get zeros so the
    summariser's force outputs are the ones consumed downstream.
    """
    metrics: Dict[str, dict] = {}
    raw_deltas: Dict[str, dict] = {}

    for functional in all_dft_functionals:
        per_system = []
        skipped_systems = 0

        for struct_name, entries in displaced.items():
            ds_sp = sp_dsets.get(struct_name)
            if ds_sp is None:
                continue

            ref_e, ref_g, dft_e, dft_g = [], [], [], []
            n_atoms = None
            for entry_name, _mol, _meta in entries:
                de_dft, dg_dft = qcf.fetch_sp_energy_gradient(
                    ds_sp, entry_name, functional,
                )
                de_ref, dg_ref = qcf.fetch_sp_energy_gradient(
                    ds_sp, entry_name, reference_spec_name,
                )
                if dg_dft is None or dg_ref is None:
                    continue
                if n_atoms is None:
                    n_atoms = dg_dft.shape[0]
                # Energies aren't part of the NM benchmark — zero them.
                ref_e.append(0.0)
                ref_g.append(dg_ref)
                dft_e.append(0.0)
                dft_g.append(dg_dft)

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
                f"  {functional}: no complete NM data — excluded."
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
                f"  {functional}: {skipped_systems} systems had no complete records."
            )
    return metrics, raw_deltas


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_nm_sampling(
    *, config, client, odset_dict: dict,
    all_dft_functionals: List[str], dft_geom_functionals: dict,
    n_adsorbate_atoms: int, res_folder: Path,
    logger: logging.Logger,
):
    """Top-level driver for the normal-mode sampling benchmark.

    Returns ``(metrics, raw_deltas)``: ``metrics`` is the per-method
    summary dict (``rmsd_force`` etc.), ``raw_deltas`` is the
    per-method concatenated delta arrays used by downstream plotting.
    """
    # Lowercase the functional spec list — same convention as the
    # geom_benchmark trajectory analysis (be_hess / energy_benchmark also
    # do this). qcf-astrochem stores specs lowercase and lookups are
    # case-sensitive.
    all_dft_functionals = [s.lower() for s in all_dft_functionals]
    reference_spec_name = config.reference_grad_lot  # already lowercased by validator
    all_spec_names = all_dft_functionals + [reference_spec_name]

    padded_log(logger, "NM-sampling analysis (DFT vs CCSD(T) gradients)")

    # 1. Collect optimised geometries
    logger.info("\nCollecting equilibrium geometries…")
    ref_mols = collect_optimized_geometries(
        odset_dict, config.geometry_opt_lot, logger,
    )
    if not ref_mols:
        logger.warning("\n  No usable equilibrium geometries — aborting.\n")
        return None, {}

    # 2. Submit Hessians + wait
    padded_log(logger, "Hessian computations")
    logger.info(
        f"\nSubmitting Hessians at {config.hessian_lot} to tag "
        f"'{config.tag_hessian}'…"
    )
    hess_ids = submit_system_hessians(
        client, ref_mols,
        config.hessian_lot, config.hessian_program,
        config.hessian_keywords, config.tag_hessian, logger,
    )
    if hess_ids:
        logger.info("\nWaiting for Hessian completion…")
        qcf.check_jobs_status(client, hess_ids, logger)

    # 3. Fetch normal modes + classify
    padded_log(logger, "Normal-mode classification")
    mode_data = collect_system_normal_modes(
        client, ref_mols, config.hessian_lot,
        n_adsorbate_atoms=n_adsorbate_atoms,
        inter_threshold=config.inter_threshold,
        bend_max_cm=config.bend_max_cm,
        logger=logger,
    )
    if not mode_data:
        logger.warning("\n  No usable normal-mode data — aborting.\n")
        return None, {}

    # 3a. Persist visualisation artifacts (always — written BEFORE the
    # imaginary-mode check so the user can open the .molden file to see
    # which modes are imaginary and decide whether to re-optimise).
    molden_dir = res_folder / "molden"
    modes_json_dir = res_folder / "json_data"
    molden_dir.mkdir(parents=True, exist_ok=True)
    modes_json_dir.mkdir(parents=True, exist_ok=True)
    for sysname, data in mode_data.items():
        mol = data["mol"]
        positions_bohr = np.asarray(mol.geometry).reshape(-1, 3)
        write_molden(
            molden_dir / f"{sysname}.molden",
            symbols=list(mol.symbols),
            geometry_bohr=positions_bohr,
            frequencies_cm=data["frequencies_cm"],
            modes_cart=data["modes_cart"],
            title=f"BEEP nm_sampling — {sysname} ({config.hessian_lot})",
        )
        write_modes_json(
            modes_json_dir / f"normal_modes_{sysname}.json",
            symbols=list(mol.symbols),
            geometry_bohr=positions_bohr,
            frequencies_cm=data["frequencies_cm"],
            modes_cart=data["modes_cart"],
            classes=data["classes"],
            n_adsorbate_atoms=n_adsorbate_atoms,
            level_of_theory=config.hessian_lot,
        )
    logger.info(
        f"\n  Normal-mode visualisations written to:\n"
        f"    {molden_dir}/   (one .molden per system)\n"
        f"    {modes_json_dir}/  (normal_modes_<system>.json)\n"
    )

    # 3b. Imaginary-mode sanity check. A genuine imaginary mode (|imag| above
    # the noise threshold) means the geometry is a saddle, not a minimum —
    # displacements around it would be meaningless. Abort unless the user
    # explicitly opted in.
    imag_offenders = {}
    for sysname, data in mode_data.items():
        freqs = data["frequencies_cm"]
        n_imag = int(np.sum(np.abs(np.imag(freqs)) > config.freq_max_imag_cm))
        if n_imag > 0:
            imag_offenders[sysname] = n_imag
    if imag_offenders:
        msg_lines = [
            f"  {s}: {n} imaginary mode(s) with |imag| > "
            f"{config.freq_max_imag_cm} cm⁻¹"
            for s, n in imag_offenders.items()
        ]
        if config.allow_imaginary_modes:
            logger.warning(
                "\n  WARNING: imaginary frequencies detected — proceeding "
                "because allow_imaginary_modes=true:\n  " + "\n  ".join(msg_lines)
                + "\n"
            )
        else:
            logger.error(
                "\n  ABORT: imaginary frequencies detected at the equilibrium "
                "geometry — the system is a saddle, not a minimum:\n  "
                + "\n  ".join(msg_lines)
                + f"\n  Open {molden_dir}/<system>.molden to see which "
                "modes are imaginary."
                + "\n  Re-optimise the geometry before nm_sampling, or set "
                "allow_imaginary_modes=true to override.\n"
            )
            raise RuntimeError(
                f"nm_sampling aborted: imaginary modes in "
                f"{list(imag_offenders)}"
            )

    # 3c. Pre-run exit. The user just wants to inspect the modes before
    # committing the CCSD(T) gradient budget; nothing past this point gets
    # submitted. Re-running with pre_run=false picks up where this left off
    # (Hessians dedup, no recompute).
    if config.pre_run:
        padded_log(
            logger,
            "Pre-run complete: modes computed, no gradient SPs submitted",
            padding_char=gear,
        )
        logger.info(
            f"\n  Inspect the modes in {molden_dir}/ before re-running with "
            "pre_run=false.\n"
        )
        return mode_data, {}

    # 4. Select picks + generate displaced molecules
    padded_log(logger, "Displacement generation")
    displaced = build_displaced_molecules(
        mode_data, dict(config.bands),
        freq_max_imag_cm=config.freq_max_imag_cm,
        extra_amplitudes_lowest_count=config.extra_amplitudes_lowest_count,
        extra_amplitude_factor=config.extra_amplitude_factor,
        logger=logger,
    )
    if not displaced:
        logger.warning("\n  No displacements generated — aborting.\n")
        return None, {}

    # 5. Build SP datasets
    logger.info("\nBuilding nm_sampling singlepoint datasets…")
    sp_dsets = build_nm_sp_datasets(
        client, displaced,
        all_dft_functionals=all_dft_functionals,
        reference_spec_name=reference_spec_name,
        reference_grad_lot=config.reference_grad_lot,
        reference_grad_program=config.reference_grad_program,
        reference_grad_keywords=config.reference_grad_keywords,
        dft_program=config.dft_program,
        dft_keyword=config.dft_keyword,
        logger=logger,
    )

    # 6. Submit
    padded_log(logger, "Submitting NM gradient SPs")
    submit_nm_singlepoints(
        sp_dsets, all_dft_functionals, reference_spec_name,
        config.tag_dft_grad, config.tag_reference_grad, logger,
    )

    # 7. Wait
    logger.info("\nWaiting for NM gradient SPs to complete…")
    wait_for_nm_completion(
        sp_dsets, all_spec_names, wait_interval=200, logger=logger,
    )

    # 8. Compute metrics
    logger.info("\nComputing per-method NM force metrics…")
    metrics, raw_deltas = compute_per_method_nm_metrics(
        displaced, sp_dsets, all_dft_functionals,
        reference_spec_name, logger,
    )

    padded_log(logger, "NM-SAMPLING BENCHMARK RESULTS", padding_char=gear)
    padded_log(logger, "NM-SAMPLING FORCE RMSE (meV/Å)")
    # Single-metric benchmark — no z-score combination needed; pass
    # ranking_df=None so the per-group winners are picked by raw RMSD_F
    # (identical order to z-scored RMSD_F, but with a more honest label).
    log_trajectory_metrics_per_group(
        logger, metrics, dft_geom_functionals, ranking_df=None,
    )
    logger.info("")
    logger.info("  Note — RMSD_F is the root-mean-square error of the per-")
    logger.info("         Cartesian-component DFT force vs the CCSD(T) reference")
    logger.info("         force, aggregated across every displaced geometry and")
    logger.info("         every binding site:")
    logger.info("")
    logger.info("           RMSD_F = sqrt( mean( (F_DFT_xyz - F_CCSDT_xyz)^2 ) )")
    logger.info("")
    logger.info("         Per-method raw values and displacement metadata are")
    logger.info("         available in:")
    logger.info("")
    logger.info("           json_data/results_nm_sampling.json   (per-method metrics)")
    logger.info("           json_data/raw_deltas_nm_sampling.npz  (per-method flat")
    logger.info("                                                  arrays of force")
    logger.info("                                                  + energy deltas)")
    logger.info("")

    # Persist JSON
    folder_path_json = res_folder / "json_data"
    folder_path_json.mkdir(parents=True, exist_ok=True)
    metrics_json = {m: {k: v for k, v in d.items()} for m, d in metrics.items()}
    (folder_path_json / "results_nm_sampling.json").write_text(
        json.dumps(metrics_json, indent=4, default=str)
    )

    # Persist raw per-functional force/energy delta arrays for downstream
    # histogram / distribution analysis. Each functional contributes two
    # flat arrays — `<functional>__force` (meV/Å, per Cartesian component)
    # and `<functional>__energy` (meV/atom). Load with
    # ``data = np.load(path)``; iterate with ``data.files``.
    deltas_path = folder_path_json / "raw_deltas_nm_sampling.npz"
    npz_payload = {}
    for functional, d in raw_deltas.items():
        npz_payload[f"{functional}__force"] = np.asarray(
            d["delta_force_meV_per_A"]
        )
        npz_payload[f"{functional}__energy"] = np.asarray(
            d["delta_e_per_atom_meV"]
        )
    np.savez_compressed(deltas_path, **npz_payload)
    logger.info(
        f"\n  NM-sampling metrics + raw deltas written to {folder_path_json}/\n"
    )

    return metrics, raw_deltas
