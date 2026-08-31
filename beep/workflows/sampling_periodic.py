"""BEEP `sampling_periodic` workflow — MLP binding-site sampling on periodic slabs.

Single-pass workflow (no refinement): for each slab in the surface collection,
place adsorbate candidates on a periodic-aware xy grid, optimize each with a
MACE MLP under periodic boundary conditions, then report per-cluster unique
binding sites via RMSD filtering.

The cell + pbc are passed to the MACE harness via the QC spec keywords (see
QCEngine's MACE harness patch that reads keywords['cell'] / keywords['pbc']).
Bottom-layer freezing goes through geomeTRIC's constraints keyword.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

from qcportal import PortalClient as FractalClient

from ..models.sampling_periodic import SamplingPeriodicConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import beep_banner
from ..adapters import qcfractal_adapter as qcf
from ..core.periodic_sampler import (
    ANG2BOHR,
    build_freeze_constraints,
    frozen_atom_indices,
    run_periodic_sampling,
    strip_adsorbate,
)
from ..core.sampling import filter_binding_sites

bcheck = "✔"
POLL_FREQUENCY_SEC = 120


welcome_msg = beep_banner(
    "Periodic Set-of-clusters Sampling",
    quote="Everywhere is here.",
    quote_author="Buckminster Fuller",
    tagline="Cover the periodic footprint, one grid node at a time.",
    authors="Stefan Vogt-Geisse and Giulia M. Bovolenta",
)


def config_summary_msg(config: SamplingPeriodicConfig) -> str:
    """Format a clean summary of the periodic sampling configuration."""
    separator = "-" * 88
    lot = config.sampling_level_of_theory
    freeze_desc = (
        f"below z={config.freeze_below_z_ang} A" if config.freeze_below_z_ang is not None
        else (f"{len(config.freeze_atoms)} atoms (explicit)" if config.freeze_atoms else "none")
    )
    lines = [
        "",
        separator,
        f"  Adsorbate:            {config.molecule}",
        f"  Surface collection:   {config.surface_collection}",
        f"  Level of theory:      {lot.display}",
        f"  PBC:                  {config.pbc}",
        f"  Step size:            {config.step_size_ang} A  (noise ±{config.grid_noise_frac*config.step_size_ang:.2f} A)",
        f"  Sampling distance:    {config.sampling_distance_ang} A",
        f"  Sanity min distance:  {config.sanity_min_distance_ang} A  (max {config.sanity_max_iter} attempts)",
        f"  Cavity z-scan:        step {config.cavity_z_scan_step_ang} A, window ±{config.cavity_z_scan_window_ang} A",
        f"  RMSD threshold:       {config.rmsd_value} A",
        f"  Freeze:               {freeze_desc}",
        f"  Random seed:          {config.random_seed}",
        separator,
        "",
    ]
    return "\n".join(lines)


def _resolve_cell(config: SamplingPeriodicConfig, surface) -> list:
    """Return the 3x3 cell (Angstrom) for a slab, preferring config over surface extras."""
    if config.cell is not None:
        return config.cell
    extras_cell = (surface.extras or {}).get("cell")
    if extras_cell is None:
        raise ValueError(
            "sampling_periodic: no cell available. Either set 'cell' in the workflow "
            "config or store it on each slab's molecule.extras['cell']."
        )
    return extras_cell


def _build_sampling_spec(
    lot,
    cell_ang,
    pbc,
    freeze_indices_0based,
    base_opt_keywords,
    logger,
):
    """Construct the OptimizationDataset spec dict for a periodic MACE run."""
    # QC spec keywords: cell + pbc for the MACE harness (see QCEngine patch)
    qc_keywords = {
        "cell": [list(row) for row in cell_ang],
        "pbc": list(pbc),
    }

    # Optimizer keywords: base defaults + user overrides + optional freeze block
    opt_keywords = {"maxiter": 125}
    if base_opt_keywords:
        opt_keywords.update(base_opt_keywords)
    freeze_constraints = build_freeze_constraints(freeze_indices_0based)
    if freeze_constraints is not None:
        existing = opt_keywords.get("constraints")
        if existing:
            if not isinstance(existing, dict):
                raise ValueError(
                    "sampling_opt_keywords['constraints'] must be geomeTRIC's JSON "
                    "form, e.g. {'freeze': [{'type': 'xyz', 'indices': [0, 1]}]}; got "
                    f"{type(existing).__name__}. The classic '$freeze / xyz 1-3' text "
                    "block is only accepted by geomeTRIC's file interface, not by the "
                    "JSON API that QCEngine drives."
                )
            logger.info(
                "  freeze constraint requested but 'constraints' already set in "
                "sampling_opt_keywords — merging freeze block into user-supplied constraints."
            )
            merged = {k: list(v) for k, v in existing.items()}
            merged.setdefault("freeze", []).extend(freeze_constraints["freeze"])
            opt_keywords["constraints"] = merged
        else:
            opt_keywords["constraints"] = freeze_constraints
        logger.info(f"  freezing {len(freeze_indices_0based)} atoms during optimization")

    spec = {
        "name": lot.lot_name,
        "description": f"Periodic sampling with {lot.display}",
        "optimization_spec": {"program": "geometric", "keywords": opt_keywords},
        "qc_spec": {
            "driver": "gradient",
            "method": lot.qc_method,
            "basis": lot.qc_basis,
            "keywords": qc_keywords,
            "program": lot.qc_program,
        },
    }
    return spec


def run(config: SamplingPeriodicConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    smol_name = config.molecule

    # Output folder: <cwd>/<molecule>/
    res_folder = Path.cwd() / smol_name
    res_folder.mkdir(parents=True, exist_ok=True)
    data_folder = res_folder / "data"
    data_folder.mkdir(exist_ok=True)

    # Per-workflow log file
    log_file = res_folder / f"sampling_periodic_{smol_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Copy input config for reproducibility
    (res_folder / f"sampling_periodic_{smol_name}.json").write_text(safe_config_dump(config))

    logger.info(welcome_msg)
    logger.info(config_summary_msg(config))

    lot = config.sampling_level_of_theory

    # RNG (seed once, share across grid noise + rotations)
    rng = random.Random(config.random_seed) if config.random_seed is not None else random.Random()

    # --- Load adsorbate ---
    ds_sm = qcf.get_collection(client, "OptimizationDataset", config.small_molecule_collection)
    try:
        # No opt-LOT for MLP-only runs; take the initial-molecule slot
        adsorbate = qcf.fetch_initial_molecule(ds_sm, smol_name, lot.lot_name)
    except KeyError:
        adsorbate = qcf.fetch_atom_molecule(client, config.atoms_collection, smol_name)

    # --- Load surfaces ---
    ds_surf = qcf.get_collection(client, "OptimizationDataset", config.surface_collection)
    all_slabs = list(ds_surf.entry_names)
    slabs = (
        [s for s in all_slabs if s in config.surface_clusters]
        if config.surface_clusters else all_slabs
    )
    logger.info(f"  Surface slabs to sample: {len(slabs)}/{len(all_slabs)}  ({', '.join(slabs)})\n")

    total_candidates = 0
    total_unique = 0

    for c, slab_name in enumerate(slabs):
        logger.info("\n" + "=" * 80)
        logger.info(f"  Slab {c+1}/{len(slabs)}: {slab_name}")
        logger.info("=" * 80 + "\n")

        surface = qcf.fetch_initial_molecule(ds_surf, slab_name, lot.lot_name)
        cell_ang = _resolve_cell(config, surface)

        # Build the OptimizationDataset for this slab's sampling run
        opt_dset_name = f"{smol_name}_{slab_name}"
        ds_opt = qcf.get_or_create_opt_dataset(client, opt_dset_name)

        # Choose freeze list (surface atoms only, indexed 0..N_surface-1)
        freeze_list = frozen_atom_indices(
            surface.geometry,
            config.freeze_below_z_ang,
            config.freeze_atoms,
            n_surface_atoms=len(surface.symbols),
        )

        # Register spec + submit
        spec = _build_sampling_spec(
            lot, cell_ang, config.pbc, freeze_list,
            config.sampling_opt_keywords, logger,
        )
        qcf.add_opt_specification(ds_opt, spec, overwrite=False)

        # Generate candidates
        candidates, debug_mol = run_periodic_sampling(
            surface=surface,
            adsorbate=adsorbate,
            cell_ang=cell_ang,
            pbc=config.pbc,
            step_size_ang=config.step_size_ang,
            grid_noise_frac=config.grid_noise_frac,
            sampling_distance_ang=config.sampling_distance_ang,
            cavity_z_scan_step_ang=config.cavity_z_scan_step_ang,
            cavity_z_scan_window_ang=config.cavity_z_scan_window_ang,
            sanity_min_distance_ang=config.sanity_min_distance_ang,
            sanity_max_iter=config.sanity_max_iter,
            rng=rng,
            logger=logger,
        )

        # Aggregate xyz (slab + every accepted adsorbate copy in its ORIGINAL
        # placement, so the sampling coverage is visible) — always written.
        aggregate_path = data_folder / f"all_sampled_sites_{slab_name}.xyz"
        debug_mol.to_file(str(aggregate_path), "xyz")
        # Per-candidate centered xyz — only when explicitly asked for.
        if config.store_initial_structures:
            debug_dir = data_folder / "site_finder" / smol_name / slab_name
            debug_dir.mkdir(parents=True, exist_ok=True)
            for name, mol in candidates:
                mol.to_file(str(debug_dir / f"{name}.xyz"), "xyz")

        # Add entries + submit
        added_names = []
        existing_names = set(ds_opt.entry_names)
        for name, mol in candidates:
            entry_name = f"{slab_name}_{name}"
            if entry_name in existing_names:
                added_names.append(entry_name)
                continue
            try:
                qcf.add_opt_entry(ds_opt, entry_name, mol)
                added_names.append(entry_name)
            except KeyError as e:
                logger.info(f"  {e}")

        if added_names:
            comp_rec = qcf.submit_optimizations(
                ds_opt, lot.lot_name, tag=config.sampling_tag, subset=added_names,
            )
            logger.info(
                f"  Submitted: {comp_rec.n_inserted} new, "
                f"{comp_rec.n_existing} already computed."
            )

        pid_list = qcf.get_job_ids(ds_opt, added_names, lot.lot_name)
        if pid_list:
            logger.info(f"  Optimizing {len(pid_list)} candidates (tag='{config.sampling_tag}')")
            qcf.wait_for_completion(client, pid_list, POLL_FREQUENCY_SEC, logger)

        # Pull optimized molecules + RMSD dedup
        opt_molecules = qcf.fetch_opt_molecules(
            ds_opt, added_names, lot.lot_name, status="COMPLETE",
        )
        n_complete = len(opt_molecules)
        logger.info(
            f"  {n_complete} optimizations COMPLETE, "
            f"{len(added_names) - n_complete} in other states."
        )

        unique = filter_binding_sites(
            opt_molecules, [], cut_off_val=config.rmsd_value,
            rmsd_symm=config.rmsd_symmetry, ligand_size=len(adsorbate.symbols),
            logger=logger, grid=0.5, nb_radius=4, dm_tau=1e-3,
        )
        n_unique = len(unique)

        # Write per-slab unique-sites list to data/
        unique_names = {name for name, _ in unique}
        report_lines = ["entry_name,is_unique"]
        for name, _ in opt_molecules:
            report_lines.append(f"{name},{'yes' if name in unique_names else 'no'}")
        (data_folder / f"unique_sites_{slab_name}.csv").write_text(
            "\n".join(report_lines) + "\n"
        )

        # --- Bare-surface companion: one MLP opt per unique confirmed site ---
        # Same LOT, same freeze policy, same cell/pbc as the complex opt.
        # Strip the adsorbate from each unique optimized complex; the
        # remaining surface positions carry the site-specific deformation.
        # Re-optimising from that state gives a physically clean bare-surface
        # reference for the BE (each site gets its own reference; no shared
        # bare slab). Entry names match the sampling entries exactly (1:1).
        n_surface_atoms = len(surface.symbols)
        surface_dset_name = f"{opt_dset_name}_surface"
        ds_surface = qcf.get_or_create_opt_dataset(client, surface_dset_name)
        qcf.add_opt_specification(ds_surface, spec, overwrite=False)

        surface_added = []
        existing_surface = set(ds_surface.entry_names)
        for entry_name, complex_mol in unique:
            if entry_name in existing_surface:
                surface_added.append(entry_name)
                continue
            bare = strip_adsorbate(complex_mol, n_surface_atoms)
            try:
                qcf.add_opt_entry(ds_surface, entry_name, bare)
                surface_added.append(entry_name)
            except KeyError as e:
                logger.info(f"  bare-surface add: {e}")

        if surface_added:
            comp_rec = qcf.submit_optimizations(
                ds_surface, lot.lot_name, tag=config.sampling_tag, subset=surface_added,
            )
            logger.info(
                f"  Bare-surface submitted: {comp_rec.n_inserted} new, "
                f"{comp_rec.n_existing} already computed."
            )

        surface_pids = qcf.get_job_ids(ds_surface, surface_added, lot.lot_name)
        if surface_pids:
            logger.info(
                f"  Optimizing {len(surface_pids)} bare-surface references "
                f"(tag='{config.sampling_tag}')"
            )
            qcf.wait_for_completion(client, surface_pids, POLL_FREQUENCY_SEC, logger)

        surface_complete = qcf.fetch_opt_molecules(
            ds_surface, surface_added, lot.lot_name, status="COMPLETE",
        )
        n_surface_complete = len(surface_complete)
        logger.info(
            f"  Bare-surface: {n_surface_complete}/{len(surface_added)} COMPLETE."
        )

        logger.info(
            f"\n  {bcheck} Slab {slab_name}: {n_complete} complex opts, "
            f"{n_unique} unique (RMSD {config.rmsd_value} A), "
            f"{n_surface_complete} bare-surface refs"
        )
        total_candidates += n_complete
        total_unique += n_unique

    logger.info("\n" + "=" * 80)
    logger.info(f"  DONE — {total_candidates} optimizations across {len(slabs)} slabs, "
                f"{total_unique} unique binding sites total.")
    logger.info("=" * 80 + "\n")
