"""BEEP be_assemble_periodic — extract per-site periodic BEs from be_comp_periodic output.

For each slab, reads the two paired specs (MACE electronic + explicit
dispersion) from the SinglepointDatasets that ``be_comp_periodic``
populates::

    <smol>_<slab>_be_sp              (SPs on optimized complex geometries)
    <smol>_<slab>_surface_be_sp      (SPs on per-site optimized bare-surface geometries)
    <smol>_gas_be_sp                 (SP on gas-phase adsorbate)

Sums electronic + dispersion energies per record to get the total BE-LOT
energy, then::

    BE_kcal = (E(complex) - E(bare_site) - E(adsorbate_gas)) * hartree2kcal
    BE_ZPVE = BE_kcal + zpve_correction_kcal_mol

Writes ``<molecule>/data/<prefix>_<slab>.csv`` per slab and a
``<prefix>_summary.csv`` across all slabs, plus a summary log line.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import qcelemental

from qcportal import PortalClient as FractalClient

from ..models.be_assemble_periodic import BeAssemblePeriodicConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import beep_banner
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import _split_dispersion

HARTREE2KCAL = qcelemental.constants.hartree2kcalmol
bcheck = "✔"


welcome_msg = beep_banner(
    "Periodic Binding-Energy Assembly",
    quote="The whole is greater than the sum of its parts.",
    quote_author="Aristotle",
    tagline="One site, one bare surface, one BE.",
    authors="Stefan Vogt-Geisse",
)


def config_summary_msg(config: BeAssemblePeriodicConfig) -> str:
    separator = "-" * 88
    lines = [
        "",
        separator,
        f"  Adsorbate:            {config.molecule}",
        f"  Slabs:                {len(config.surface_clusters)}  ({', '.join(config.surface_clusters)})",
        f"  BE electronic LOT:    {config.be_electronic_lot.display}",
        f"  BE dispersion:        {config.be_dispersion}",
        f"  ZPVE correction:      {config.zpve_correction_kcal_mol} kcal/mol",
        f"  Output prefix:        {config.output_prefix}",
        separator,
        "",
    ]
    return "\n".join(lines)


def _spec_names(config: BeAssemblePeriodicConfig) -> tuple:
    """Return (electronic_spec_name, dispersion_spec_name) — must match be_comp_periodic."""
    elec_alias = config.be_electronic_lot.alias
    _bare, _, _ = _split_dispersion(config.be_dispersion)
    disp_suffix = config.be_dispersion[len(_bare):]
    return elec_alias.lower(), f"{elec_alias}{disp_suffix}".lower()


def _summed_energy(ds_sp, entry_name: str, elec_spec: str, disp_spec: str) -> Optional[float]:
    """Return E_electronic + E_dispersion (hartree) or None if any piece is missing/errored."""
    e_elec, _ = qcf.fetch_sp_energy_gradient(ds_sp, entry_name, elec_spec)
    e_disp, _ = qcf.fetch_sp_energy_gradient(ds_sp, entry_name, disp_spec)
    if e_elec is None or e_disp is None:
        return None
    return float(e_elec) + float(e_disp)


def run(config: BeAssemblePeriodicConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    smol_name = config.molecule
    res_folder = Path.cwd() / smol_name
    res_folder.mkdir(parents=True, exist_ok=True)
    data_folder = res_folder / "data"
    data_folder.mkdir(exist_ok=True)

    log_file = res_folder / f"be_assemble_periodic_{smol_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    (res_folder / f"be_assemble_periodic_{smol_name}.json").write_text(safe_config_dump(config))

    logger.info(welcome_msg)
    logger.info(config_summary_msg(config))

    elec_spec, disp_spec = _spec_names(config)
    logger.info(f"  spec lookup: electronic='{elec_spec}', dispersion='{disp_spec}'")

    # --- Gas-phase adsorbate energy (once) ---
    gas_dset_name = f"{smol_name}_gas_be_sp"
    try:
        ds_gas = qcf.get_collection(client, "singlepoint", gas_dset_name)
    except Exception as e:
        logger.info(f"\nFATAL: cannot open gas-phase SP dataset {gas_dset_name}: {e}")
        logger.info("Did be_comp_periodic run for this adsorbate?")
        return

    e_gas = _summed_energy(ds_gas, smol_name, elec_spec, disp_spec)
    if e_gas is None:
        logger.info(f"\nFATAL: gas-phase energy for {smol_name} is missing/errored.")
        logger.info("Wait for be_comp_periodic to complete, then rerun.")
        return
    logger.info(f"\n  E(gas, {smol_name}) = {e_gas:.8f} Ha  ({e_gas * HARTREE2KCAL:.4f} kcal/mol)")

    # --- Per-slab assembly ---
    summary_rows = []   # (slab, entry, be_kcal, be_zpve_kcal)
    total_sites_written = 0

    for slab_name in config.surface_clusters:
        logger.info("\n" + "=" * 80)
        logger.info(f"  Slab: {slab_name}")
        logger.info("=" * 80)

        complex_dset_name = f"{smol_name}_{slab_name}_be_sp"
        surface_dset_name = f"{smol_name}_{slab_name}_surface_be_sp"
        try:
            ds_complex = qcf.get_collection(client, "singlepoint", complex_dset_name)
            ds_surface = qcf.get_collection(client, "singlepoint", surface_dset_name)
        except Exception as e:
            logger.info(f"  skip {slab_name}: {e}")
            continue

        common = sorted(set(ds_complex.entry_names) & set(ds_surface.entry_names))
        if not common:
            logger.info(f"  no common entries between {complex_dset_name} and {surface_dset_name}")
            continue

        # Header for the per-slab CSV
        rows = ["entry_name,E_complex_Ha,E_surface_Ha,E_gas_Ha,BE_kcal_mol,BE_ZPVE_kcal_mol"]
        n_ok = n_skip = 0

        for entry_name in common:
            e_complex = _summed_energy(ds_complex, entry_name, elec_spec, disp_spec)
            e_surface = _summed_energy(ds_surface, entry_name, elec_spec, disp_spec)
            if e_complex is None or e_surface is None:
                logger.info(
                    f"  skip {entry_name}: "
                    f"E_complex={'OK' if e_complex is not None else 'MISSING'}, "
                    f"E_surface={'OK' if e_surface is not None else 'MISSING'}"
                )
                n_skip += 1
                continue
            be_ha = e_complex - e_surface - e_gas
            be_kcal = be_ha * HARTREE2KCAL
            be_zpve = be_kcal + config.zpve_correction_kcal_mol
            rows.append(
                f"{entry_name},{e_complex:.8f},{e_surface:.8f},{e_gas:.8f},"
                f"{be_kcal:.4f},{be_zpve:.4f}"
            )
            summary_rows.append((slab_name, entry_name, be_kcal, be_zpve))
            n_ok += 1

        csv_path = data_folder / f"{config.output_prefix}_{slab_name}.csv"
        csv_path.write_text("\n".join(rows) + "\n")
        total_sites_written += n_ok
        logger.info(
            f"  {bcheck} {slab_name}: {n_ok} sites written, {n_skip} skipped  →  {csv_path.name}"
        )

    # --- Aggregate summary CSV ---
    summary_path = data_folder / f"{config.output_prefix}_summary.csv"
    summary_lines = ["slab,entry_name,BE_kcal_mol,BE_ZPVE_kcal_mol"]
    for slab, entry, be_kcal, be_zpve in summary_rows:
        summary_lines.append(f"{slab},{entry},{be_kcal:.4f},{be_zpve:.4f}")
    summary_path.write_text("\n".join(summary_lines) + "\n")

    logger.info("\n" + "=" * 80)
    logger.info(f"  DONE — {total_sites_written} periodic BEs across {len(config.surface_clusters)} slabs")
    logger.info(f"         summary → data/{summary_path.name}")
    logger.info("=" * 80 + "\n")
