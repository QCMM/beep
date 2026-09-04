"""BEEP be_comp_periodic — submit periodic BE single-points on sampling outputs.

Per slab, registers the range-separated MACE + explicit-dispersion pair of
SP specs on:
- ``<smol>_<slab>_be_sp``            (complex geometries from sampling_periodic)
- ``<smol>_<slab>_surface_be_sp``    (per-site bare-surface geometries)

Once (across all runs), on:
- ``<smol>_gas_be_sp``               (gas-phase adsorbate, non-periodic)

Submits everything, waits for completion. Assembly happens in
``be_assemble_periodic``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from qcportal import PortalClient as FractalClient

from ..models.be_comp_periodic import BeCompPeriodicConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import beep_banner
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import _split_dispersion, periodic_dispersion_program

bcheck = "✔"
POLL_FREQUENCY_SEC = 120


welcome_msg = beep_banner(
    "Periodic Binding-Energy Computation",
    quote="A wet sheet and a flowing sea, and a wind that follows fast.",
    quote_author="Allan Cunningham",
    tagline="Range-separated MACE meets periodic dispersion.",
    authors="Stefan Vogt-Geisse",
)


def config_summary_msg(config: BeCompPeriodicConfig) -> str:
    separator = "-" * 88
    cell_source = "config-level" if config.cell is not None else "per-slab extras"
    lines = [
        "",
        separator,
        f"  Adsorbate:            {config.molecule}",
        f"  Slabs:                {len(config.surface_clusters)}  ({', '.join(config.surface_clusters)})",
        f"  BE electronic LOT:    {config.be_electronic_lot.display}",
        f"  BE dispersion:        {config.be_dispersion}",
        f"  PBC (slab SPs):       {config.pbc}",
        f"  Cell (slab SPs):      {cell_source}",
        f"  Compute tag:          {config.be_tag}",
        separator,
        "",
    ]
    return "\n".join(lines)


def _build_be_specs(
    ds_sp,
    electronic_lot,
    be_dispersion: str,
    keywords_periodic: dict,
    keywords_gas: dict,
    logger,
    periodic: bool,
) -> Tuple[List[str], str, str]:
    """Register the paired (electronic, dispersion) BE specs on a SinglepointDataset.

    Returns (spec_names, elec_spec, disp_spec). ``keywords_periodic`` is
    used when ``periodic=True`` (adds cell + pbc); ``keywords_gas`` when
    ``periodic=False`` (empty for a gas-phase adsorbate reference).
    """
    kw = keywords_periodic if periodic else keywords_gas
    elec_alias = electronic_lot.alias  # MACE file stem
    _bare, _disp_method, disp_program = _split_dispersion(be_dispersion)
    # Route D3 to the periodic-capable harness. The legacy ``dftd3`` executable
    # wrapper silently ignores cell/pbc, so a slab would get cluster dispersion
    # with no error. Applied to the gas-phase specs too, so that the
    # BE = complex - surface - gas difference cancels within one harness.
    disp_program = periodic_dispersion_program(disp_program)
    disp_suffix = be_dispersion[len(_bare):]

    elec_spec = qcf.add_energy_spec(
        ds_sp, spec_name=elec_alias,
        method=electronic_lot.qc_method, basis=None, program="mace",
        keywords=kw,
        description=f"BE electronic ({electronic_lot.display})"
                    + (" [periodic]" if periodic else " [gas]"),
    )
    disp_spec = qcf.add_energy_spec(
        ds_sp, spec_name=f"{elec_alias}{disp_suffix}",
        method=be_dispersion, basis=None, program=disp_program,
        keywords=kw,
        description=f"BE dispersion ({be_dispersion} via {disp_program})"
                    + (" [periodic]" if periodic else " [gas]"),
    )
    logger.info(
        f"  registered specs on {ds_sp.name}: {elec_spec}  +  {disp_spec}"
    )
    return [elec_spec, disp_spec], elec_spec, disp_spec


def _submit_and_collect(
    ds_sp, spec_names: List[str], subset: List[str], tag: str, logger,
) -> List[int]:
    """Submit SPs for a subset of entries + return the resulting record IDs."""
    if not subset:
        return []
    meta = qcf.submit_singlepoints_in_dataset(ds_sp, spec_names, tag=tag, subset=subset)
    logger.info(
        f"  submit {ds_sp.name}: {meta.n_inserted} new, {meta.n_existing} existing"
    )
    pids: List[int] = []
    for spec_name in spec_names:
        for entry_name in subset:
            rec = ds_sp.get_record(entry_name, spec_name)
            if rec is not None:
                pids.append(rec.id)
    return pids


def _resolve_cell(config: BeCompPeriodicConfig, surface_extras, record_cell=None) -> list:
    """Config-level `cell` wins, then the cell the geometries were optimized
    under, then surface Molecule.extras['cell']."""
    if config.cell is not None:
        return config.cell
    if record_cell is not None:
        return record_cell
    extras_cell = (surface_extras or {}).get("cell")
    if extras_cell is None:
        raise ValueError(
            "be_comp_periodic: no cell available. Set 'cell' in the workflow config, "
            "or store it on each slab's molecule.extras['cell']; it is normally read "
            "back from the optimization spec sampling_periodic registered."
        )
    return extras_cell


def run(config: BeCompPeriodicConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    smol_name = config.molecule
    res_folder = Path.cwd() / smol_name
    res_folder.mkdir(parents=True, exist_ok=True)
    data_folder = res_folder / "data"
    data_folder.mkdir(exist_ok=True)

    log_file = res_folder / f"be_comp_periodic_{smol_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    (res_folder / f"be_comp_periodic_{smol_name}.json").write_text(safe_config_dump(config))

    logger.info(welcome_msg)
    logger.info(config_summary_msg(config))

    elec_lot = config.be_electronic_lot
    opt_lot = config.opt_level_of_theory

    # --- Gas-phase adsorbate reference (once) ---
    logger.info("\n--- gas-phase adsorbate reference ---")
    ds_sm = qcf.get_collection(client, "OptimizationDataset", config.small_molecule_collection)
    try:
        adsorbate = qcf.fetch_final_molecule(ds_sm, smol_name, elec_lot.lot_name)
    except KeyError:
        adsorbate = qcf.fetch_initial_molecule(ds_sm, smol_name, elec_lot.lot_name)
        logger.info(
            f"  {smol_name} not optimized at {elec_lot.display}; using initial geometry"
        )

    ds_gas = qcf.get_or_create_singlepoint_dataset(client, f"{smol_name}_gas_be_sp")
    gas_specs, _, _ = _build_be_specs(
        ds_gas, elec_lot, config.be_dispersion,
        keywords_periodic={}, keywords_gas={}, logger=logger, periodic=False,
    )
    existing_gas = set(ds_gas.entry_names)
    if smol_name not in existing_gas:
        qcf.add_singlepoint_entries(ds_gas, [(smol_name, adsorbate)])
    gas_pids = _submit_and_collect(
        ds_gas, gas_specs, subset=[smol_name], tag=config.be_tag, logger=logger,
    )

    # --- Per-slab BE SPs on complex + bare-surface geometries ---
    all_pids: List[int] = list(gas_pids)

    for c, slab_name in enumerate(config.surface_clusters):
        logger.info("\n" + "=" * 80)
        logger.info(f"  Slab {c+1}/{len(config.surface_clusters)}: {slab_name}")
        logger.info("=" * 80)

        complex_dset_name = f"{smol_name}_{slab_name}"
        surface_dset_name = f"{complex_dset_name}_surface"
        try:
            ds_complex = qcf.get_collection(client, "OptimizationDataset", complex_dset_name)
            ds_surface = qcf.get_collection(client, "OptimizationDataset", surface_dset_name)
        except Exception as e:
            logger.info(f"  skip {slab_name}: {e}")
            continue

        # Only work on entries that exist in BOTH datasets (bare exists only
        # for RMSD-unique confirmed sites from sampling_periodic).
        complex_entries = set(ds_complex.entry_names)
        surface_entries = set(ds_surface.entry_names)
        common = sorted(complex_entries & surface_entries)
        if not common:
            logger.info(f"  no entries common to {complex_dset_name} and {surface_dset_name}; skip")
            continue

        # Pull the final optimized molecules for each; use surface.extras for cell fallback
        # (any complete surface record works — they all sit on the same slab).
        surface_final = qcf.fetch_opt_molecules(
            ds_surface, common, opt_lot, status="COMPLETE",
        )
        complex_final = qcf.fetch_opt_molecules(
            ds_complex, common, opt_lot, status="COMPLETE",
        )
        surface_final_map = dict(surface_final)
        complex_final_map = dict(complex_final)
        complete_common = [n for n in common if n in surface_final_map and n in complex_final_map]

        if not complete_common:
            logger.info(f"  no COMPLETE sites common to both datasets; skip {slab_name}")
            continue

        # Cell: config-level or from any slab record's extras
        sample_mol = surface_final_map[complete_common[0]]
        record_cell, _ = qcf.fetch_opt_cell(ds_surface, complete_common[0], opt_lot)
        cell_ang = _resolve_cell(config, sample_mol.extras, record_cell)
        keywords_periodic = {
            "cell": [list(row) for row in cell_ang],
            "pbc": list(config.pbc),
        }
        logger.info(
            f"  {len(complete_common)}/{len(common)} sites COMPLETE in both datasets"
        )

        # Register + submit on complex SP dataset
        ds_complex_sp = qcf.get_or_create_singlepoint_dataset(
            client, f"{complex_dset_name}_be_sp",
        )
        specs_complex, _, _ = _build_be_specs(
            ds_complex_sp, elec_lot, config.be_dispersion,
            keywords_periodic=keywords_periodic, keywords_gas={},
            logger=logger, periodic=True,
        )
        existing = set(ds_complex_sp.entry_names)
        new_entries = [
            (n, complex_final_map[n]) for n in complete_common if n not in existing
        ]
        if new_entries:
            qcf.add_singlepoint_entries(ds_complex_sp, new_entries)
        pids_c = _submit_and_collect(
            ds_complex_sp, specs_complex, subset=complete_common,
            tag=config.be_tag, logger=logger,
        )
        all_pids.extend(pids_c)

        # Register + submit on bare-surface SP dataset
        ds_surface_sp = qcf.get_or_create_singlepoint_dataset(
            client, f"{surface_dset_name}_be_sp",
        )
        specs_surface, _, _ = _build_be_specs(
            ds_surface_sp, elec_lot, config.be_dispersion,
            keywords_periodic=keywords_periodic, keywords_gas={},
            logger=logger, periodic=True,
        )
        existing = set(ds_surface_sp.entry_names)
        new_entries = [
            (n, surface_final_map[n]) for n in complete_common if n not in existing
        ]
        if new_entries:
            qcf.add_singlepoint_entries(ds_surface_sp, new_entries)
        pids_s = _submit_and_collect(
            ds_surface_sp, specs_surface, subset=complete_common,
            tag=config.be_tag, logger=logger,
        )
        all_pids.extend(pids_s)

        logger.info(
            f"  {bcheck} slab {slab_name}: submitted {len(pids_c) + len(pids_s)} SPs "
            f"({len(complete_common)} sites x 2 specs x 2 datasets)"
        )

    # --- Wait for the whole set ---
    if all_pids:
        logger.info(f"\nWaiting on {len(all_pids)} SP records (tag='{config.be_tag}')")
        qcf.wait_for_completion(client, all_pids, POLL_FREQUENCY_SEC, logger)

    logger.info("\n" + "=" * 80)
    logger.info(f"  DONE — be_comp_periodic submitted + polled {len(all_pids)} records.")
    logger.info("=" * 80 + "\n")
