"""BEEP be_assemble_periodic — extract per-site periodic BE / IE / DE from be_comp_periodic output.

Per slab, reads the ReactionDatasets ``be_comp_periodic`` populates::

    <smol>_<slab>_be     complex - relaxed bare surface - gas-phase adsorbate
    <smol>_<slab>_ie     complex - slab - adsorbate (fragments frozen at the complex geometry)
    <smol>_<slab>_de     frozen slab + frozen adsorbate - relaxed bare surface - gas adsorbate

each with the paired electronic (MACE) and dispersion reaction specs; a site's energy is the sum
of the two. With ``quantity`` 'be' (or 'all') writes ``<molecule>/data/<prefix>_<slab>.csv`` and
``<prefix>_summary.csv`` with::

    BE_kcal  = total_energy(_be) * hartree2kcal
    BE_ZPVE  = BE_kcal + zpve_correction_kcal_mol

and with 'ie' (or 'all') ``<prefix>_ie_<slab>.csv`` and ``<prefix>_ie_summary.csv`` with the IE
(no ZPVE). With 'all' these also carry the BE and the deformation energy DE (the ``_de``
reaction), split into the slab part E(slab_frozen) - E(bare_site) and the adsorbate part
E(ads_frozen) - E(adsorbate_gas); BE = IE + DE is checked per site.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


def interaction_rows(
    e_complex: Dict[str, float],
    e_slab_frozen: Dict[str, float],
    e_ads_frozen: Dict[str, float],
    e_surface: Optional[Dict[str, float]] = None,
    e_gas: Optional[Union[float, Dict[str, float]]] = None,
) -> List[Tuple]:
    """Per-site IE (and, given the relaxed references, BE and its deformation split), kcal/mol.

    Returns tuples ``(entry, E_complex, E_slab_frozen, E_ads_frozen, IE)`` or, when
    ``e_surface`` and ``e_gas`` are given, ``(... , IE, BE, DE_slab, DE_ads, DE)`` for the
    sites present in every mapping. BE = IE + DE_slab + DE_ads holds by construction.
    ``e_gas`` is one energy or a per-site mapping.
    """
    with_be = e_surface is not None and e_gas is not None
    keys = set(e_complex) & set(e_slab_frozen) & set(e_ads_frozen)
    if with_be:
        keys &= set(e_surface)
        if isinstance(e_gas, dict):
            keys &= set(e_gas)
    rows = []
    for n in sorted(keys):
        ec, es, ea = e_complex[n], e_slab_frozen[n], e_ads_frozen[n]
        ie = (ec - es - ea) * HARTREE2KCAL
        if not with_be:
            rows.append((n, ec, es, ea, ie))
            continue
        eg = e_gas[n] if isinstance(e_gas, dict) else e_gas
        de_slab = (es - e_surface[n]) * HARTREE2KCAL
        de_ads = (ea - eg) * HARTREE2KCAL
        be = (ec - e_surface[n] - eg) * HARTREE2KCAL
        rows.append((n, ec, es, ea, ie, be, de_slab, de_ads, de_slab + de_ads))
    return rows


# Component roles per reaction kind, in the order of ``split_components``: components are
# matched by the sign of their coefficient and, within a sign, by size (slab > adsorbate).
ROLES = {
    "be": {+1: ["complex"], -1: ["surface", "gas"]},
    "ie": {+1: ["complex"], -1: ["slab_frozen", "ads_frozen"]},
    "de": {+1: ["slab_frozen", "ads_frozen"], -1: ["surface", "gas"]},
}


def split_components(comps: List[Tuple[float, int, float]], kind: str) -> Optional[Dict[str, float]]:
    """Energies by role from the ``(coefficient, n_atoms, energy)`` components of one periodic
    BE/IE/DE reaction record (see ``ROLES``), or None if the record has another shape."""
    out: Dict[str, float] = {}
    for sign, roles in ROLES[kind].items():
        part = sorted((c for c in comps if (c[0] > 0) == (sign > 0)), key=lambda c: -c[1])
        if len(part) != len(roles) or any(abs(c[0]) != 1.0 for c in part):
            return None
        if len(part) == 2 and part[0][1] == part[1][1]:
            return None
        for role, c in zip(roles, part):
            out[role] = c[2]
    return out


def reaction_energies(ds_rxn, kind: str, names, elec_spec: str, disp_spec: str,
                      logger) -> Dict[str, Dict[str, float]]:
    """Per-site component energies by role (electronic + dispersion, hartree) from a periodic
    BE/IE/DE ReactionDataset, plus ``"total"`` = sum of coefficient x energy."""
    parts = {}
    for spec in (elec_spec, disp_spec):
        got = {}
        for name, _spec, rec in ds_rxn.iterate_records(entry_names=names, specification_names=[spec],
                                                       include=["components"]):
            comps = qcf.reaction_component_energies(rec)
            split = split_components(comps, kind) if comps is not None else None
            if split is not None:
                got[name] = split
        parts[spec] = got
    out: Dict[str, Dict[str, float]] = {}
    for n in names:
        a, b = parts[elec_spec].get(n), parts[disp_spec].get(n)
        if a is None or b is None:
            logger.info(f"  skip {n}: {kind.upper()} {'electronic' if a is None else 'dispersion'} "
                        f"reaction MISSING")
            continue
        e = {role: a[role] + b[role] for role in a}
        e["total"] = sum(sign * e[role] for sign, roles in ROLES[kind].items() for role in roles)
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

    want_be = config.quantity in ("be", "all")
    want_ie = config.quantity in ("ie", "all")

    # --- Per-slab assembly ---
    summary_rows = []   # (slab, entry, be_kcal, be_zpve_kcal)
    total_sites_written = 0
    ie_summary_rows = []  # (slab, *interaction_rows tuple)
    total_ie_written = 0

    for slab_name in config.surface_clusters:
        logger.info("\n" + "=" * 80)
        logger.info(f"  Slab: {slab_name}")
        logger.info("=" * 80)

        base = f"{smol_name}_{slab_name}{config.dataset_suffix}"
        kinds = ["be"] * want_be + ["ie"] * want_ie + ["de"] * (want_be and want_ie)
        try:
            ds = {k: qcf.get_collection(client, "reaction", f"{base}_{k}{config.sp_dataset_suffix}")
                  for k in kinds}
        except Exception as e:
            logger.info(f"  skip {slab_name}: {e}")
            continue
        en = {k: reaction_energies(ds[k], k, sorted(ds[k].entry_names), elec_spec, disp_spec, logger)
             for k in kinds}

        if want_be:
            rows = ["entry_name,E_complex_Ha,E_surface_Ha,E_gas_Ha,BE_kcal_mol,BE_ZPVE_kcal_mol"]
            for entry_name, r in sorted(en["be"].items()):
                be_kcal = r["total"] * HARTREE2KCAL
                be_zpve = be_kcal + config.zpve_correction_kcal_mol
                rows.append(
                    f"{entry_name},{r['complex']:.8f},{r['surface']:.8f},{r['gas']:.8f},"
                    f"{be_kcal:.4f},{be_zpve:.4f}"
                )
                summary_rows.append((slab_name, entry_name, be_kcal, be_zpve))
            n_ok = len(rows) - 1
            csv_path = data_folder / f"{config.output_prefix}_{slab_name}.csv"
            csv_path.write_text("\n".join(rows) + "\n")
            total_sites_written += n_ok
            logger.info(
                f"  {bcheck} {slab_name}: {n_ok}/{len(ds['be'].entry_names)} BE sites written  →  {csv_path.name}"
            )

        if want_ie:
            ie = en["ie"]
            ec = {n: r["complex"] for n, r in ie.items()}
            es = {n: r["slab_frozen"] for n, r in ie.items()}
            ea = {n: r["ads_frozen"] for n, r in ie.items()}
            e_surf = e_gas = None
            if want_be:
                # sites with all three reactions; DE from its own record must equal BE - IE
                both = set(en["be"]) & set(en["de"])
                e_surf = {n: en["be"][n]["surface"] for n in both}
                e_gas = {n: en["be"][n]["gas"] for n in both}
                dev = [abs(en["de"][n]["total"] - (en["be"][n]["total"] - ie[n]["total"])) * HARTREE2KCAL
                       for n in both & set(ie)]
                if dev and max(dev) > 1e-6:
                    logger.info(f"  WARNING {slab_name}: |DE - (BE - IE)| up to {max(dev):.2e} kcal/mol")
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
                f"  {bcheck} {slab_name}: {len(rows_ie)}/{len(ds['ie'].entry_names)} IE sites written  →  {ie_path.name}"
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
