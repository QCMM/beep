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

With ``quantity`` 'ie' or 'all', also reads the frozen fragments::

    <smol>_<slab>_ie_slab_sp         (slab at the complex geometry, adsorbate removed)
    <smol>_<slab>_ie_ads_sp          (adsorbate at the complex geometry, isolated)

and writes ``<prefix>_ie_<slab>.csv`` and ``<prefix>_ie_summary.csv`` with::

    IE_kcal = (E(complex) - E(slab_frozen) - E(ads_frozen)) * hartree2kcal

(no ZPVE). With 'all' these files also carry the BE and the deformation energy
DE = BE - IE, split into the slab part E(slab_frozen) - E(bare_site) and the
adsorbate part E(ads_frozen) - E(adsorbate_gas).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        f"  Quantity:             {config.quantity}",
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


def interaction_rows(
    e_complex: Dict[str, float],
    e_slab_frozen: Dict[str, float],
    e_ads_frozen: Dict[str, float],
    e_surface: Optional[Dict[str, float]] = None,
    e_gas: Optional[float] = None,
) -> List[Tuple]:
    """Per-site IE (and, given the relaxed references, BE and its deformation split), kcal/mol.

    Returns tuples ``(entry, E_complex, E_slab_frozen, E_ads_frozen, IE)`` or, when
    ``e_surface`` and ``e_gas`` are given, ``(... , IE, BE, DE_slab, DE_ads, DE)`` for the
    sites present in every mapping. BE = IE + DE_slab + DE_ads holds by construction.
    """
    with_be = e_surface is not None and e_gas is not None
    keys = set(e_complex) & set(e_slab_frozen) & set(e_ads_frozen)
    if with_be:
        keys &= set(e_surface)
    rows = []
    for n in sorted(keys):
        ec, es, ea = e_complex[n], e_slab_frozen[n], e_ads_frozen[n]
        ie = (ec - es - ea) * HARTREE2KCAL
        if not with_be:
            rows.append((n, ec, es, ea, ie))
            continue
        de_slab = (es - e_surface[n]) * HARTREE2KCAL
        de_ads = (ea - e_gas) * HARTREE2KCAL
        be = (ec - e_surface[n] - e_gas) * HARTREE2KCAL
        rows.append((n, ec, es, ea, ie, be, de_slab, de_ads, de_slab + de_ads))
    return rows


def _energies(ds_sp, names, elec_spec: str, disp_spec: str, logger, label: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for n in names:
        e = _summed_energy(ds_sp, n, elec_spec, disp_spec)
        if e is None:
            logger.info(f"  skip {n}: E_{label} MISSING")
        else:
            out[n] = e
    return out


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
    want_be = config.quantity in ("be", "all")
    want_ie = config.quantity in ("ie", "all")
    e_gas = None
    if want_be:
        gas_dset_name = f"{smol_name}_gas_be_sp{config.dataset_suffix}"
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
    ie_summary_rows = []  # (slab, *interaction_rows tuple)
    total_ie_written = 0

    for slab_name in config.surface_clusters:
        logger.info("\n" + "=" * 80)
        logger.info(f"  Slab: {slab_name}")
        logger.info("=" * 80)

        complex_dset_name = f"{smol_name}_{slab_name}{config.dataset_suffix}_be_sp{config.sp_dataset_suffix}"
        surface_dset_name = f"{smol_name}_{slab_name}{config.dataset_suffix}_surface_be_sp{config.sp_dataset_suffix}"
        try:
            ds_complex = qcf.get_collection(client, "singlepoint", complex_dset_name)
            ds_surface = (qcf.get_collection(client, "singlepoint", surface_dset_name)
                          if want_be else None)
        except Exception as e:
            logger.info(f"  skip {slab_name}: {e}")
            continue

        common = []
        if want_be:
            common = sorted(set(ds_complex.entry_names) & set(ds_surface.entry_names))
            if not common:
                logger.info(f"  no common entries between {complex_dset_name} and {surface_dset_name}")
                if not want_ie:
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

        if want_ie:
            base = f"{smol_name}_{slab_name}{config.dataset_suffix}"
            try:
                ds_slab_fz = qcf.get_collection(
                    client, "singlepoint", f"{base}_ie_slab_sp{config.sp_dataset_suffix}")
                ds_ads_fz = qcf.get_collection(
                    client, "singlepoint", f"{base}_ie_ads_sp{config.sp_dataset_suffix}")
            except Exception as e:
                logger.info(f"  skip IE for {slab_name}: {e}")
                continue
            ie_names = sorted(set(ds_complex.entry_names) & set(ds_slab_fz.entry_names)
                              & set(ds_ads_fz.entry_names))
            ec = _energies(ds_complex, ie_names, elec_spec, disp_spec, logger, "complex")
            es = _energies(ds_slab_fz, ie_names, elec_spec, disp_spec, logger, "slab_frozen")
            ea = _energies(ds_ads_fz, ie_names, elec_spec, disp_spec, logger, "ads_frozen")
            e_surf = None
            if want_be:
                e_surf = _energies(ds_surface, [n for n in common if n in ec],
                                   elec_spec, disp_spec, logger, "surface")
            rows_ie = interaction_rows(ec, es, ea, e_surf, e_gas)
            header = "entry_name,E_complex_Ha,E_slab_frozen_Ha,E_ads_frozen_Ha,IE_kcal_mol"
            if want_be:
                header += ",BE_kcal_mol,DE_slab_kcal_mol,DE_ads_kcal_mol,DE_kcal_mol"
            lines = [header]
            for r in rows_ie:
                lines.append(",".join([r[0]] + [f"{x:.8f}" for x in r[1:4]]
                                      + [f"{x:.4f}" for x in r[4:]]))
                ie_summary_rows.append((slab_name,) + tuple(r))
            ie_path = data_folder / f"{config.output_prefix}_ie_{slab_name}.csv"
            ie_path.write_text("\n".join(lines) + "\n")
            total_ie_written += len(rows_ie)
            logger.info(
                f"  {bcheck} {slab_name}: {len(rows_ie)}/{len(ie_names)} IE sites written  →  {ie_path.name}"
            )

    # --- Aggregate summary CSV ---
    if want_be:
        summary_path = data_folder / f"{config.output_prefix}_summary.csv"
        summary_lines = ["slab,entry_name,BE_kcal_mol,BE_ZPVE_kcal_mol"]
        for slab, entry, be_kcal, be_zpve in summary_rows:
            summary_lines.append(f"{slab},{entry},{be_kcal:.4f},{be_zpve:.4f}")
        summary_path.write_text("\n".join(summary_lines) + "\n")
    if want_ie:
        ie_summary_path = data_folder / f"{config.output_prefix}_ie_summary.csv"
        header = "slab,entry_name,IE_kcal_mol"
        if want_be:
            header += ",BE_kcal_mol,DE_slab_kcal_mol,DE_ads_kcal_mol,DE_kcal_mol"
        lines = [header]
        for r in ie_summary_rows:
            lines.append(",".join([r[0], r[1]] + [f"{x:.4f}" for x in r[5:]]))
        ie_summary_path.write_text("\n".join(lines) + "\n")

    logger.info("\n" + "=" * 80)
    if want_be:
        logger.info(f"  DONE — {total_sites_written} periodic BEs across {len(config.surface_clusters)} slabs")
        logger.info(f"         summary → data/{summary_path.name}")
    if want_ie:
        logger.info(f"  DONE — {total_ie_written} periodic IEs across {len(config.surface_clusters)} slabs")
        logger.info(f"         summary → data/{ie_summary_path.name}")
    logger.info("=" * 80 + "\n")
