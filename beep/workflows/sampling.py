"""Sampling workflow — refactored from workflows/launch_sampling.py."""
import json
import logging
from pathlib import Path

from qcfractal.interface.client import FractalClient

from ..models.sampling import SamplingConfig
from ..core.logging_utils import padded_log
from ..core.errors import DatasetNotFound, LevelOfTheoryNotFound
from ..adapters import qcfractal_adapter as qcf

bcheck = "\u2714"
gear = "\u2699"
separator = "-" * 80

welcome_msg = f"""
{separator}
Welcome to the BEEP  Set-of-clusters Sampling Workflow
{separator}

"Adopt the pace of nature: her secret is patience."

                              \u2013 Ralph Waldo Emerson

Seek Locate Map
{separator}

                            By:  Stefan Vogt-Geisse and Giulia M. Bovolenta
"""


def config_summary_msg(config):
    """Format a clean summary of the sampling configuration."""
    s_lot = config.sampling_level_of_theory
    r_lot = config.refinement_level_of_theory
    s_basis = s_lot.basis or "N/A"
    lines = [
        "",
        separator,
        f"  Molecule:             {config.molecule}",
        f"  Surface model:        {config.surface_model_collection}",
        f"  Small molecule coll:  {config.small_molecule_collection}",
        f"  Sampling LOT:         {s_lot.method}/{s_basis} ({s_lot.program})",
        f"  Refinement LOT:       {r_lot.method}/{r_lot.basis} ({r_lot.program})",
        f"  Sampling shell:       {config.sampling_shell} Angstrom",
        f"  Sampling condition:   {config.sampling_condition}",
        f"  RMSD cutoff:          {config.rmsd_value}",
        f"  RMSD symmetry:        {config.rmsd_symmetry}",
        f"  Target binding sites: {config.total_binding_sites}",
        separator,
        "",
    ]
    return "\n".join(lines)


def check_collection_existence(client, *collections, collection_type="OptimizationDataset"):
    for collection in collections:
        if not qcf.check_collection_exists(client, collection_type, collection):
            raise DatasetNotFound(
                f"Collection {collection} does not exist. Please create it first. Exiting..."
            )


def check_optimized_molecule(ds, opt_lot, mol_names):
    for mol in list(mol_names):
        try:
            rr = qcf.fetch_opt_record(ds, mol, opt_lot)
        except KeyError:
            raise LevelOfTheoryNotFound(
                f"{opt_lot} level of theory for {mol} or the entry itself does not exist "
                f"in {ds.name} collection. Add the molecule and optimize it first\n"
            )
        if rr.status == "INCOMPLETE":
            raise ValueError(f" Optimization has status {rr.status} restart it or wait")
        elif rr.status == "ERROR":
            raise ValueError(f" Optimization has status {rr.status} restart it or wait")


def process_refinement(client, ropt_lot_name, rmethod, rbasis, program,
                       qc_keyword, ds_opt, logger, rtag="refinement"):
    spec = {
        "name": ropt_lot_name,
        "description": f"Geometric + {rmethod}/{rbasis}/{program}",
        "optimization_spec": {"program": "geometric", "keywords": None},
        "qc_spec": {
            "driver": "gradient",
            "method": rmethod,
            "basis": rbasis,
            "keywords": qc_keyword,
            "program": program,
        },
    }

    qcf.add_opt_specification(ds_opt, spec, overwrite=True)
    ds_opt.save()
    c = qcf.submit_optimizations(ds_opt, ropt_lot_name, tag=rtag)

    logger.info(
        f"\nRefinement optimization initiated with specification '{ropt_lot_name}' \n"
        f"using {rmethod}/{rbasis} in {program}. \n"
        f"Description: {spec['description']}. \n"
        f"Tag applied: '{rtag}'\n"
        f"Number of optimizations submitted: {c}. {bcheck} \n"
    )


def run(config: SamplingConfig, client: FractalClient) -> None:
    from beep.adapters.qcfractal_adapter import run_sampling

    logger = logging.getLogger("beep")

    smol_name = config.molecule

    # Create output folder: <molecule>/sampling/
    res_folder = Path.cwd() / smol_name / "sampling"
    res_folder.mkdir(parents=True, exist_ok=True)

    # File logging inside the output folder
    log_file = res_folder / f"beep_sampling_{smol_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Save a copy of the input config
    config_path = res_folder / f"sampling_{smol_name}.json"
    config_path.write_text(json.dumps(config.dict(), indent=4, default=str))

    logger.info(welcome_msg)

    method = config.sampling_level_of_theory.method
    basis = config.sampling_level_of_theory.basis
    program = config.sampling_level_of_theory.program
    rmethod = config.refinement_level_of_theory.method
    rbasis = config.refinement_level_of_theory.basis
    rprogram = config.refinement_level_of_theory.program

    qc_keyword = config.keyword_id

    if basis:
        opt_lot = method + "_" + basis
    else:
        opt_lot = method
    ropt_lot = rmethod + "_" + rbasis

    # --- Configuration summary ---
    logger.info(config_summary_msg(config))

    args_dict = {
        "method": method,
        "basis": basis,
        "program": program,
        "tag": config.sampling_tag,
        "kw_id": None,  # keyword_id is used for refinement spec, not sampling
        "rmsd_symm": config.rmsd_symmetry,
        "store_initial": config.store_initial_structures,
        "rmsd_val": config.rmsd_value,
        "sampling_shell": config.sampling_shell,
        "sampling_condition": config.sampling_condition,
        "opt_lot": opt_lot,
        "logger": logger,
    }

    # --- Validate collections ---
    logger.info("Validating collections and optimized geometries...")
    check_collection_existence(
        client, config.surface_model_collection, config.small_molecule_collection
    )
    ds_sm = qcf.get_collection(client, "OptimizationDataset", config.small_molecule_collection)
    ds_wc = qcf.get_collection(client, "OptimizationDataset", config.surface_model_collection)

    cluster_names = list(ds_wc.data.records.keys())
    logger.info(f"  Surface model clusters: {len(cluster_names)}  ({', '.join(cluster_names)})")

    # Check if all the molecules are optimized at the requested level of theory
    if len(qcf.fetch_initial_molecule(ds_sm, smol_name, opt_lot).symbols) == 1:
        check_optimized_molecule(ds_wc, opt_lot, ds_wc.data.records.keys())
        args_dict["target_mol"] = qcf.fetch_initial_molecule(ds_sm, smol_name, opt_lot)
    else:
        check_optimized_molecule(ds_sm, opt_lot, [smol_name])
        check_optimized_molecule(ds_wc, opt_lot, ds_wc.data.records.keys())
        args_dict["target_mol"] = qcf.fetch_final_molecule(ds_sm, smol_name, opt_lot)

    logger.info(f"  All geometries validated. {bcheck}\n")

    args_dict["client"] = client

    # --- Sampling loop ---
    count = 0
    cluster_results = []

    for c, w in enumerate(ds_wc.data.records):
        args_dict["cluster"] = qcf.fetch_final_molecule(ds_wc, w, opt_lot)

        ref_opt_dset_name = smol_name + "_" + w
        smplg_opt_dset_name = "pre_" + ref_opt_dset_name

        ds_smplg = qcf.get_or_create_opt_dataset(client, smplg_opt_dset_name)
        ds_ref = qcf.get_or_create_opt_dataset(client, ref_opt_dset_name)
        args_dict["sampling_opt_dset"] = ds_smplg
        args_dict["refinement_opt_dset"] = ds_ref

        logger.info(f"\n{'=' * 80}")
        logger.info(f"  Cluster {c+1}/{len(cluster_names)}: {w}")
        logger.info(f"  Sampling dataset:    {smplg_opt_dset_name}")
        logger.info(f"  Refinement dataset:  {ref_opt_dset_name}")
        logger.info(f"{'=' * 80}\n")

        debug_path = res_folder / "site_finder" / (str(smol_name) + "_w") / w
        if not debug_path.exists() and config.store_initial_structures:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
        args_dict["debug_path"] = debug_path

        run_sampling(**args_dict)

        process_refinement(
            client, ropt_lot, rmethod, rbasis, rprogram,
            qc_keyword, ds_ref, logger, config.refinement_tag,
        )

        ds_ref = qcf.get_or_create_opt_dataset(client, ref_opt_dset_name)
        n_sites = len(ds_ref.data.records)
        count += n_sites
        cluster_results.append((w, n_sites))

        logger.info(f"\n  {bcheck} Cluster {w}: {n_sites} binding sites  |  Running total: {count}")

        if count > config.total_binding_sites:
            logger.info(f"\n  Target of {config.total_binding_sites} binding sites reached. Stopping early.")
            break

    # --- Final summary ---
    logger.info(f"\n\n{'=' * 80}")
    logger.info(f"  SAMPLING SUMMARY FOR {smol_name}")
    logger.info(f"{'=' * 80}")
    logger.info(f"  {'Cluster':<15} {'Binding sites':>15}")
    logger.info(f"  {'-'*15} {'-'*15}")
    for w, n in cluster_results:
        logger.info(f"  {w:<15} {n:>15}")
    logger.info(f"  {'-'*15} {'-'*15}")
    logger.info(f"  {'TOTAL':<15} {count:>15}")
    logger.info(f"{'=' * 80}\n")

    logger.removeHandler(file_handler)
    file_handler.close()
