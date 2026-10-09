"""BEEP be_comp_periodic — submit periodic BE / IE / DE on sampling outputs.

Per slab, one ReactionDataset per quantity, each entry (one per site) carrying its own
stoichiometry (:func:`beep.core.stoichiometry.periodic_stoichiometry`):
- ``<smol>_<slab>_be``   complex - relaxed bare surface - gas-phase adsorbate
- ``<smol>_<slab>_ie``   complex - slab - adsorbate (fragments frozen at the complex geometry)
- ``<smol>_<slab>_de``   frozen slab + frozen adsorbate - relaxed bare surface - gas adsorbate
``quantity`` 'be' builds ``_be``, 'ie' builds ``_ie`` (no bare surface or gas-phase reference
needed), 'all' builds all three. With ``surface_family`` set, the slabs share one dataset per
quantity, ``<smol>_<family>_be`` etc. (one specification: common lateral cell, non-periodic axis
padded to the largest slab's; entry names carry the slab). Every component is evaluated with the same periodic
specification (cell, pbc), as the range-separated pair of an electronic (MACE) and a
dispersion reaction specification; a component shared between datasets is one record.

Submits everything, waits for completion. Assembly happens in
``be_assemble_periodic``.
"""
from __future__ import annotations

import logging

import numpy as np
from beep.core.periodic_sampler import (
    filter_periodic_sites,
    pad_nonperiodic_axes,
)
from beep.core.stoichiometry import periodic_stoichiometry

BOHR2ANG = 0.529177210903
from pathlib import Path
from typing import Dict, List, Tuple

from qcportal import PortalClient as FractalClient

from qcportal.reaction import ReactionDatasetNewEntry

from ..core.entry_guard import check_entry_geometry
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
        f"  Quantity:             {config.quantity}"
        + (f" (sites: {config.ie_site_filter})" if config.quantity == "ie" else ""),
        f"  Datasets:             <mol>_{config.surface_family or '<slab>'}{config.dataset_suffix}_{{{','.join(_quantity_kinds(config.quantity))}}}"
        f"{config.sp_dataset_suffix}",
        f"  PBC (slab SPs):       {config.pbc}",
        f"  Cell (slab SPs):      {cell_source}",
        f"  Compute tags:         {config.be_tag} (electronic), {config.disp_tag or config.be_tag} (dispersion)",
        separator,
        "",
    ]
    return "\n".join(lines)


def _quantity_kinds(quantity: str) -> List[str]:
    """ReactionDataset suffixes built for a ``quantity``."""
    return {"be": ["be"], "ie": ["ie"], "all": ["be", "ie", "de"]}[quantity]


def _build_reaction_specs(ds_rxn, kind: str, electronic_lot, be_dispersion: str,
                          keywords_periodic: dict, logger) -> List[str]:
    """Register the paired (electronic, dispersion) reaction specs on a periodic BE/IE/DE
    ReactionDataset. The spec names are the electronic alias and alias + dispersion suffix;
    both singlepoint specs carry the periodic keywords."""
    elec_alias = electronic_lot.alias
    _bare, _disp_method, disp_program = _split_dispersion(be_dispersion)
    # Route D3 to the periodic-capable harness. The legacy ``dftd3`` executable
    # wrapper silently ignores cell/pbc, so a slab would get cluster dispersion
    # with no error.
    disp_program = periodic_dispersion_program(disp_program)
    disp_suffix = be_dispersion[len(_bare):]
    label = kind.upper()
    elec = qcf.add_reaction_energy_spec(
        ds_rxn, spec_name=elec_alias, method=electronic_lot.qc_method, basis=None, program="mace",
        keywords=keywords_periodic, description=f"{label} electronic ({electronic_lot.display}) [periodic]",
    )
    disp = qcf.add_reaction_energy_spec(
        ds_rxn, spec_name=f"{elec_alias}{disp_suffix}", method=be_dispersion, basis=None,
        program=disp_program, keywords=keywords_periodic,
        description=f"{label} dispersion ({be_dispersion} via {disp_program}) [periodic]",
    )
    logger.info(f"  registered reaction specs on {ds_rxn.name}: {elec}  +  {disp}")
    return [elec, disp]


def _component_key(coefficient: float, mol) -> Tuple[float, int]:
    return (float(coefficient), len(mol.symbols))


def _guard_reused_reactions(ds_rxn, stoich: Dict[str, list], cell_ang, pbc) -> None:
    """A reused entry must hold the same components: same coefficients and, component by
    component (matched by coefficient and size), the same geometry modulo lattice vectors."""
    reused = [n for n in ds_rxn.entry_names if n in stoich]
    if not reused:
        return
    for entry in ds_rxn.iterate_entries(entry_names=reused):
        old = sorted(((x.coefficient, x.molecule) for x in entry.stoichiometries),
                     key=lambda cm: _component_key(*cm))
        new = sorted(((c, m) for m, c in stoich[entry.name]), key=lambda cm: _component_key(*cm))
        if [_component_key(*cm) for cm in old] != [_component_key(*cm) for cm in new]:
            raise ValueError(
                f"{ds_rxn.name}/{entry.name}: an entry of this name already exists with a different "
                f"stoichiometry. Give this run its own datasets with 'dataset_suffix' (e.g. '_v1')."
            )
        for (_, m_old), (_, m_new) in zip(old, new):
            check_entry_geometry(m_old, m_new, entry.name, ds_rxn.name, cell_ang=cell_ang, pbc=pbc)


def _submit_reactions(ds_rxn, stoich: Dict[str, list], spec_tags: Dict[str, str],
                      cell_ang, pbc, logger) -> List[int]:
    """Add the missing entries (``stoich``: name -> [(molecule, coefficient)]), submit every
    entry for each spec of ``spec_tags`` (spec name -> compute tag) and return the reaction
    record IDs."""
    _guard_reused_reactions(ds_rxn, stoich, cell_ang, pbc)
    existing = set(ds_rxn.entry_names)
    new_entries = [ReactionDatasetNewEntry(name=n, stoichiometries=[(c, m) for m, c in st])
                   for n, st in stoich.items() if n not in existing]
    if new_entries:
        qcf._check_insert_meta(ds_rxn.add_entries(new_entries), f"entries in {ds_rxn.name}")
    names = sorted(stoich)
    for tag in sorted(set(spec_tags.values())):
        specs = [s for s, t in spec_tags.items() if t == tag]
        meta = ds_rxn.submit(entry_names=names, specification_names=specs, compute_tag=tag)
        logger.info(f"  submit {ds_rxn.name} {specs} -> {tag}: {meta.n_inserted} new, {meta.n_existing} existing")
    pids: List[int] = []
    for spec_name in spec_tags:
        for n in names:
            rec = ds_rxn.get_record(n, spec_name)
            if rec is not None:
                pids.append(rec.id)
    return pids


def common_cell(slab_jobs, pbc) -> list:
    """One cell for the reactions of several slabs: they must share the periodic axes (within
    1e-6 A); each non-periodic axis takes the largest padded length over the slabs (any length
    above the slab's own padding is equally valid there)."""
    cells = [np.asarray(cell, dtype=float) for _, _, _, cell in slab_jobs]
    out = cells[0].copy()
    for k in range(3):
        if pbc[k]:
            for (slab, _, _, _), c in zip(slab_jobs, cells):
                if np.abs(c[k] - out[k]).max() > 1e-6:
                    raise ValueError(f"surface_family: slab {slab} has another periodic cell vector {k} "
                                     f"({c[k].tolist()} vs {out[k].tolist()}); slabs of one family must share "
                                     f"the lateral cell")
        else:
            out[k, k] = max(float(c[k, k]) for c in cells)
    return [list(row) for row in out]


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


def gas_adsorbate(ds_sm, smol_name: str, elec_lot, logger):
    """The adsorbate optimized at the electronic LOT, else the entry's input geometry.

    The collection need not carry a spec for the electronic LOT at all (qcportal then raises
    PortalRequestError, not KeyError), so the spec is checked before asking for the record;
    the fallback reads the entry itself, which needs no spec.
    """
    if elec_lot.lot_name in ds_sm.specifications:
        try:
            return qcf.fetch_final_molecule(ds_sm, smol_name, elec_lot.lot_name)
        except KeyError:
            pass
    logger.info(f"  {smol_name} not optimized at {elec_lot.display}; using initial geometry")
    return qcf.fetch_entry_initial_molecule(ds_sm, smol_name)


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

    # --- Gas-phase adsorbate (BE/DE reference; its charge and multiplicity also set the
    #     frozen adsorbate fragment's) ---
    logger.info("\n--- gas-phase adsorbate reference ---")
    ds_sm = qcf.get_collection(client, "OptimizationDataset", config.small_molecule_collection)
    adsorbate = gas_adsorbate(ds_sm, smol_name, elec_lot, logger)

    all_pids: List[int] = []
    slab_jobs = []          # (slab, complex dataset, stoichiometries, padded cell)
    n_ads = len(adsorbate.symbols)

    # --- Per slab: complexes, bare surfaces (be/all) -> BE / IE / DE reactions ---
    for c, slab_name in enumerate(config.surface_clusters):
        logger.info("\n" + "=" * 80)
        logger.info(f"  Slab {c+1}/{len(config.surface_clusters)}: {slab_name}")
        logger.info("=" * 80)

        complex_dset_name = f"{smol_name}_{slab_name}{config.dataset_suffix}"
        surface_dset_name = f"{complex_dset_name}_surface"
        want_be = config.quantity in ("be", "all")
        try:
            ds_complex = qcf.get_collection(client, "OptimizationDataset", complex_dset_name)
            ds_surface = (qcf.get_collection(client, "OptimizationDataset", surface_dset_name)
                          if want_be else None)
        except Exception as e:
            logger.info(f"  skip {slab_name}: {e}")
            continue

        if want_be:
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
        else:
            # quantity='ie': no bare-surface dataset; the sites are the complete complexes,
            # optionally deduplicated with the same periodic filter sampling_periodic uses.
            names = sorted(ds_complex.entry_names)
            complex_final_map = dict(qcf.fetch_opt_molecules(ds_complex, names, opt_lot, status="COMPLETE"))
            complete_common = sorted(complex_final_map)
            if not complete_common:
                logger.info(f"  no COMPLETE complexes in {complex_dset_name}; skip {slab_name}")
                continue

        # Cell: config-level or from any slab record's extras
        cell_source = ds_surface if want_be else ds_complex
        sample_mol = (surface_final_map if want_be else complex_final_map)[complete_common[0]]
        record_cell, _ = qcf.fetch_opt_cell(cell_source, complete_common[0], opt_lot)
        cell_ang = _resolve_cell(config, sample_mol.extras, record_cell)
        # Pad non-periodic axes. The cell recorded by sampling_periodic can be
        # thinner than the slab (X x X x X/2 against an 18 A slab), and the
        # dispersion backends wrap along every axis regardless of the pbc mask,
        # which folds the adsorbate into the slab. Padding here also repairs the
        # BE for geometries optimized before this was understood, without
        # re-running the optimizations, whose MACE energies were unaffected.
        complex_geom = np.asarray(
            complex_final_map[complete_common[0]].geometry, dtype=float
        ).reshape(-1, 3) * BOHR2ANG
        cell_ang = pad_nonperiodic_axes(cell_ang, config.pbc, complex_geom)
        logger.info(
            f"  cell for SPs (non-periodic axes padded): "
            f"{[round(float(cell_ang[i][i]), 2) for i in range(3)]} Angstrom"
        )
        if want_be:
            logger.info(
                f"  {len(complete_common)}/{len(common)} sites COMPLETE in both datasets"
            )
        elif config.ie_site_filter == "unique":
            energies = qcf.fetch_opt_energies(ds_complex, complete_common, opt_lot)
            unique = filter_periodic_sites(
                [(n, complex_final_map[n]) for n in complete_common], cell_ang, config.pbc,
                n_adsorbate_atoms=n_ads, com_tol_ang=config.com_tol_ang,
                orient_tol_ang=config.orientation_tol_ang, energies=energies, logger=logger,
            )
            n_before = len(complete_common)
            complete_common = sorted(name for name, _ in unique)
            logger.info(f"  IE sites: {len(complete_common)} unique of {n_before} complete complexes")
        else:
            logger.info(f"  IE sites: all {len(complete_common)} complete complexes")

        # each entry carries its stoichiometry; submitted below, per slab or per surface family
        n_slab = len(complex_final_map[complete_common[0]].symbols) - n_ads
        stoich = {
            n: periodic_stoichiometry(
                complex_final_map[n], n_slab,
                surface_mol=surface_final_map[n] if want_be else None, gas_mol=adsorbate,
            )
            for n in complete_common
        }
        slab_jobs.append((slab_name, complex_dset_name, stoich, cell_ang))
        logger.info(f"  {bcheck} slab {slab_name}: {len(complete_common)} sites (quantity={config.quantity})")

    # --- Submit: one ReactionDataset per quantity, per slab or for the whole surface family ---
    if config.surface_family:
        groups = [(f"{smol_name}_{config.surface_family}{config.dataset_suffix}",
                   common_cell(slab_jobs, config.pbc),
                   {n: st for _, _, stoich, _ in slab_jobs for n, st in stoich.items()})] if slab_jobs else []
    else:
        groups = [(dset, cell, stoich) for _, dset, stoich, cell in slab_jobs]
    for base, cell_ang, stoich in groups:
        keywords_periodic = {"cell": [list(row) for row in cell_ang], "pbc": list(config.pbc)}
        n_records = 0
        for kind in _quantity_kinds(config.quantity):
            ds_rxn = qcf.create_reaction_dataset(client, f"{base}_{kind}{config.sp_dataset_suffix}")
            specs = _build_reaction_specs(
                ds_rxn, kind, elec_lot, config.be_dispersion, keywords_periodic, logger,
            )
            elec_spec, disp_spec = specs
            pids = _submit_reactions(
                ds_rxn, {n: st[kind] for n, st in stoich.items()},
                {elec_spec: config.be_tag, disp_spec: config.disp_tag or config.be_tag},
                cell_ang=cell_ang, pbc=config.pbc, logger=logger,
            )
            n_records += len(pids)
            all_pids.extend(pids)
        logger.info(f"  {bcheck} {base}: submitted {n_records} reactions ({len(stoich)} sites, "
                    f"cell {[round(float(cell_ang[i][i]), 2) for i in range(3)]} A)")

    # --- Wait for the whole set ---
    if all_pids:
        logger.info(f"\nWaiting on {len(all_pids)} reaction records (tags '{config.be_tag}', "
                    f"'{config.disp_tag or config.be_tag}')")
        qcf.wait_for_completion(client, all_pids, POLL_FREQUENCY_SEC, logger)

    logger.info("\n" + "=" * 80)
    logger.info(f"  DONE — be_comp_periodic submitted + polled {len(all_pids)} records.")
    logger.info("=" * 80 + "\n")
