"""BE + Hessian workflow — refactored from workflows/launch_be_hess.py."""
import json
import logging
from pathlib import Path
from typing import List, Tuple

import qcportal as ptl
from qcportal.client import FractalClient
from qcportal.collections import ReactionDataset

from ..models.be_hess import BeHessConfig
from ..core.logging_utils import padded_log
from ..adapters import qcfractal_adapter as qcf

bcheck = "\u2714"
gear = "\u2699"

welcome_msg = """
---------------------------------------------------------------------------------------
Welcome to the BEEP Binding Energy and Hessian Computation Suite
---------------------------------------------------------------------------------------

"Schoenheit ist der Glanz der Wahrheit"

                  ~ Werner Heissenberg

---------------------------------------------------------------------------------------

                            By:  Stefan Vogt-Geisse and Giulia M. Bovolenta

"""


def check_collections(client, surface_model_name, molecule_collection_name,
                       molecule_name, optimization_level):
    logger = logging.getLogger("beep")
    try:
        molecule_dataset = qcf.get_collection(client, "OptimizationDataset", molecule_collection_name)
    except KeyError:
        logger.info(f"Collection {molecule_collection_name} with the target molecules does not exist, please create it first.")
        raise

    try:
        initial_mol = qcf.fetch_initial_molecule(molecule_dataset, molecule_name, optimization_level)
        if len(initial_mol.symbols) == 1:
            final_molecule = initial_mol
        else:
            final_molecule = qcf.fetch_final_molecule(molecule_dataset, molecule_name, optimization_level)
    except KeyError:
        logger.info(f"{molecule_name} is not optimized at the requested level of theory, please optimize them first.")
        raise

    try:
        surface_dataset = qcf.get_collection(client, "OptimizationDataset", surface_model_name)
    except KeyError:
        logger.info(f"Collection with set of clusters that span the surface {surface_model_name} does not exist. Please create it first.")
        raise

    logger.info(f"Successfully extracted {surface_dataset.name} and molecule {final_molecule.name} {bcheck}")
    return surface_dataset, final_molecule


def get_optdataset(client, surf_ds, mol_name, opt_lot, exclude_clusters=None):
    if exclude_clusters is None:
        exclude_clusters = []
    processed_datasets = []
    logger = logging.getLogger("beep")

    for cn in surf_ds.df.index:
        ds_opt_name = f"{mol_name}_{cn}"
        if cn in exclude_clusters:
            continue

        try:
            opt_ds = qcf.get_collection(client, "OptimizationDataset", ds_opt_name)
            if not opt_ds.data.records:
                raise ValueError(f"Dataset '{ds_opt_name}' exists but contains no entries.")
        except Exception as e:
            logger.info(f"Warning: Error accessing dataset '{ds_opt_name}': {e}, might not exist. Will continue without")
            continue

        qcf.wait_for_dataset_completion(client, opt_ds, opt_lot, logger, wait_interval=60)
        processed_datasets.append(opt_ds)

    logger.info(f"Finished optimization datasets: {[d.name for d in processed_datasets]}\n\n\n")
    return processed_datasets


def process_be_computation(client, logger, finished_opt_list, surf_opt_ds,
                           smol_mol, opt_lot, mult, config):
    all_ids = []
    opt_method = opt_lot.split("_")[0]
    opt_basis = opt_lot.split("_")[1]

    for ds_opt in finished_opt_list:
        padded_log(logger, f"Checking {ds_opt.name} for repeated structures", padding_char="*", total_length=60)
        opt_stru = qcf.rmsd_filter_from_dataset(ds_opt, opt_lot, logger)

        padded_log(logger, f"Building name for the new ReactionDataset", padding_char="*", total_length=60)
        cluster_name = "_".join(list(opt_stru.keys())[0].split("_")[-3:-1])
        cluster_mol = qcf.fetch_final_molecule(surf_opt_ds, cluster_name, opt_lot)
        rdset_name = f"be_{config.molecule.upper()}_{cluster_name.upper()}_{opt_method.upper()}_{opt_basis.upper()}"
        logger.info(f"ReactionDataset name for {ds_opt.name} is: {rdset_name}")

        padded_log(logger, f"Creating the dataset {rdset_name}", padding_char="*", total_length=60)
        ds_be = qcf.create_or_load_reaction_dataset(
            client, rdset_name, opt_lot, smol_mol, cluster_mol, ds_opt, opt_stru, logger
        )

        keyword = None
        if mult == 2:
            keyword_obj = ptl.models.KeywordSet(values={"reference": "uks"})
            keyword = "rad_be"
            try:
                ds_be.add_keywords(keyword, config.program, keyword_obj, default=True)
                ds_be.save()
            except KeyError:
                pass

        padded_log(logger, f"Sending computations for {rdset_name}", padding_char="*", total_length=60)
        job_ids = qcf.compute_be_dft_energies(
            ds_be, config.level_of_theory, config.energy_tag,
            program=config.program, keyword=keyword, logger=logger,
        )
        all_ids.extend(job_ids)
        logger.info(f"Finished processing {rdset_name}\n\n\n")

    return all_ids


def run(config: BeHessConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    # Create output folder: <molecule>/be_hess/
    res_folder = Path.cwd() / config.molecule / "be_hess"
    res_folder.mkdir(parents=True, exist_ok=True)

    # File logging inside the output folder
    log_file = res_folder / f"beep_be_hess_{config.molecule}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Save a copy of the input config
    config_path = res_folder / f"be_hess_{config.molecule}.json"
    config_path.write_text(json.dumps(config.dict(), indent=4, default=str))

    logger.info(welcome_msg)

    padded_log(logger, f"Checking for the state of the OptimizationDatasets")
    opt_lot = config.opt_level_of_theory
    surf_opt_ds, smol_mol = check_collections(
        client, config.surface_model_collection,
        config.small_molecule_collection, config.molecule, opt_lot,
    )

    mult = smol_mol.molecular_multiplicity
    logger.info(f"\nThe molecular multiplicity is {mult}\n")

    opt_method = opt_lot.split("_")[0]
    opt_basis = opt_lot.split("_")[1]

    # Process binding energy computations
    if config.level_of_theory:
        padded_log(logger, f"Starting computation of Binding energies")
        be_parameters = (
            f"Binding Energy Computation Parameters:\n"
            f"- Molecule: {config.molecule}\n"
            f"- Surface Model Collection: {config.surface_model_collection}\n"
            f"- Small Molecule Collection: {config.small_molecule_collection}\n"
            f"- Level of Theory: {' '.join(config.level_of_theory)}\n"
            f"- Optimization Level of Theory: {opt_lot}\n"
            f"- Program: {config.program}\n"
            f"- Energy Tag: {config.energy_tag or 'Default: energies'}\n"
        )
        logger.info(be_parameters)
        padded_log(logger, f"Extracting OptimizationDatasets with no INCOMPLETE optimization")
        finished_opt_list = get_optdataset(
            client, surf_opt_ds, config.molecule, opt_lot, config.exclude_clusters,
        )
        padded_log(logger, f"Creating {len(finished_opt_list)} ReactionDatasets for BE computation:")
        all_ids = process_be_computation(
            client, logger, finished_opt_list, surf_opt_ds, smol_mol, opt_lot, mult, config,
        )
        padded_log(logger, f"Checking for completion of ALL binding energy computations")
        qcf.check_jobs_status(client, all_ids, logger, wait_interval=600)

    # Process Hessian computations
    if config.hessian_clusters:
        hessian_parameters = (
            f"Hessian Computation Parameters:\n"
            f"- Molecule: {config.molecule}\n"
            f"- Hessian Clusters: {', '.join(config.hessian_clusters)}\n"
            f"- Optimization Level of Theory: {opt_lot}\n"
            f"- Program: {config.program}\n"
            f"- Hessian Tag: {config.hessian_tag or 'Default: hessian'}\n"
        )
        padded_log(logger, f"Starting computation of Hessians")
        logger.info(hessian_parameters)
        all_hess_ids = []
        for cluster_name in config.hessian_clusters:
            rdset_name = f"be_{config.molecule.upper()}_{cluster_name.upper()}_{opt_method.upper()}_{opt_basis.upper()}"
            hess_ids = qcf.compute_hessian(
                client, rdset_name, opt_lot, mult, config.hessian_tag,
                logger=logger, program=config.program,
            )
            all_hess_ids.extend(hess_ids)
        padded_log(logger, f"Checking for completion of ALL Hessian computations")
        logger.info(f"The IDs for the Hessian computations are: {all_hess_ids}\n")
        qcf.check_jobs_status(client, all_hess_ids, logger, wait_interval=3600)

    logger.info("\nThank you for using the binding energy and hessian compute suite!")

    logger.removeHandler(file_handler)
    file_handler.close()
