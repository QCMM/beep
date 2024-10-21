import argparse
import logging
import qcfractal.interface as ptl
from typing import List, Tuple, Dict
from collections import Counter

from qcfractal.interface.client import FractalClient
from qcfractal.interface.collections import Dataset, OptimizationDataset, ReactionDataset
from qcelemental.models.molecule import Molecule

from beep.utils.logging_utils import *
from beep.utils.qcf_utils import *
from beep.binding_energy_compute import *

bcheck = "\u2714"
mia0911 = "\6"
gear = "\u2699"
wstar = "\u2606"
welcome_msg = """       
·······················································································
:                                                                                     :
:  ██████╗ ██╗███╗   ██╗██████╗ ██╗███╗   ██╗ ██████╗                                 :
:  ██╔══██╗██║████╗  ██║██╔══██╗██║████╗  ██║██╔════╝                                 :
:  ██████╔╝██║██╔██╗ ██║██║  ██║██║██╔██╗ ██║██║  ███╗                                :
:  ██╔══██╗██║██║╚██╗██║██║  ██║██║██║╚██╗██║██║   ██║                                :
:  ██████╔╝██║██║ ╚████║██████╔╝██║██║ ╚████║╚██████╔╝                                :
:  ╚═════╝ ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝                                 :
:                                                                                     :
:  ███████╗███╗   ██╗███████╗██████╗  ██████╗ ██╗   ██╗                               :
:  ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔════╝ ╚██╗ ██╔╝                               :
:  █████╗  ██╔██╗ ██║█████╗  ██████╔╝██║  ███╗ ╚████╔╝                                :
:  ██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██║   ██║  ╚██╔╝                                 :
:  ███████╗██║ ╚████║███████╗██║  ██║╚██████╔╝   ██║                                  :
:  ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝                                  :
:                                                                                     :
:  ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗  :
:  ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║  :
:  █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║  :
:  ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║  :
:  ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║  :
:  ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝  :
:                                                                                     :
:  ██████╗ ██╗      █████╗ ████████╗███████╗ ██████╗ ██████╗ ███╗   ███╗              :
:  ██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║              :
:  ██████╔╝██║     ███████║   ██║   █████╗  ██║   ██║██████╔╝██╔████╔██║              :
:  ██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║              :
:  ██║     ███████╗██║  ██║   ██║   ██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║              :
:  ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝              :
:                                                                                     :
·······················································································

---------------------------------------------------------------------------------------
Welcome to the BEEP Binding Energy and Hessian Computation Suite
---------------------------------------------------------------------------------------


"Schönheit ist der Glanz der Wahrheit"
                  
                  ~ Werner Heissenberg

---------------------------------------------------------------------------------------

                            By:  Stefan Vogt-Geisse and Giulia M. Bovolenta



"""

def get_arguments() -> argparse.Namespace:
    usage = """This script is used to set up various parameters for QCFractal server connections and calculations."""
    parser = argparse.ArgumentParser(description=usage)

    parser.add_argument("--client-address",
                        help="The URL address and port of the QCFractal server (default: localhost:7777)",
                        default="localhost:7777"
    )
    parser.add_argument("--username",
                        help="The username for the database client (Default = None)",
                        default=None
    )
    parser.add_argument("--password",
                        help="The password for the database client (Default = None)",
                        default=None
    )
    parser.add_argument("--surface-model-collection",
                        help="The name of the collection with the water clusters (default: Water_22)",
                        default="Water_22"
    )
    parser.add_argument("--small-molecule-collection",
                        help="The name of the collection with small molecules or radicals (default: Small_molecules)",
                        default="Small_molecules"
    )
    parser.add_argument("--molecule",
                        help="The name of the molecule for the binding energy computation"
    )
    parser.add_argument("--level-of-theory",
                        nargs='+',
                        default=[],
                        help="The level(s) of theory for the binding energy computation in the format: method_basis (default: wpbe-d3bj_def2-tzvp)",
    )
    parser.add_argument("--exclude-clusters",
                        nargs='+',
                        default=[],
                        help="Binding sites on these clusters will be excluded (Default: [])",
    )
    parser.add_argument("--opt-level-of-theory",
                        help="The level of theory of the binding sites optimization in the format: method_basis (default: hf3c_minix)",
    )
    parser.add_argument("--keyword-id",
                        help="ID of the QCfractal for the single point computations keywords (default: None)",
                        default=None
    )
    parser.add_argument("--hessian-clusters",
                        nargs='*',
                        default=[],
                        help="List of molecules for which to compute the hessian for the binding sites of the chosen cluster (default: empty list)"
    )
    parser.add_argument("-p", "--program",
                        default="psi4",
                        help="The program to use for this calculation (default: psi4)"
    )
    parser.add_argument("--energy-tag",
                        help="The tag to use to specify the qcfractal-manager for the BE computations (default: energies)"
    )
    parser.add_argument("--hessian-tag",
                        help="The tag to use to specify the qcfractal-manager for the hessian (default: hessian)"
    )

    return parser.parse_args()



def check_collections(client: ptl.FractalClient, surface_model_name: str, molecule_collection_name: str, molecule_name: str, optimization_level: str) -> Tuple[ptl.collections.OptimizationDataset, ptl.collections.OptimizationDataset, ptl.Molecule]:
    logger = logging.getLogger("beep")
    try:
        molecule_dataset = client.get_collection("OptimizationDataset", molecule_collection_name)
    except KeyError:
        logger.info(f"Collection {molecule_collection_name} with the target molecules does not exist, please create it first.")
        raise KeyError
    
    try:
        final_molecule = molecule_dataset.get_record(molecule_name, optimization_level).get_final_molecule()
    except KeyError:
        logger.info(f"{molecule_name} is not optimized at the requested level of theory, please optimize them first.")
        raise KeyError
    
    try:
        surface_dataset = client.get_collection("OptimizationDataset", surface_model_name)
    except KeyError:
        logger.info(f"Collection with set of clusters that span the surface {surface_model_name} does not exist. Please create it first.")
        raise KeyError
    logger.info(f"Successfully extracted {surface_dataset.name} and molecule {final_molecule.name} {bcheck}")
    return surface_dataset, final_molecule

def get_incomplete_entries(opt_ds: ptl.collections.OptimizationDataset, opt_lot: str) -> List[str]:
    logger = logging.getLogger("beep")
    incomplete_entries = []
    for entry in opt_ds.df.index:
        entry_status = opt_ds.get_record(entry, opt_lot)
        if entry_status == "ERROR":
            logger.info(f"Warning: Entry '{entry}' in dataset finished with ERROR")
        elif entry_status == "INCOMPLETE":
            incomplete_entries.append(entry)
    return incomplete_entries

def get_optdataset(client: ptl.FractalClient, surf_ds: ptl.collections.OptimizationDataset, mol_name: str, opt_lot: str, exclude_clusters = []) -> List[str]:
    processed_datasets = []
    logger = logging.getLogger("beep")

    for cn in surf_ds.df.index:
        ds_opt_name = f"{mol_name}_{cn}"
        if cn in exclude_clusters:
            continue

        # Attempt to retrieve the dataset and handle exceptions
        try:
            opt_ds = client.get_collection("OptimizationDataset", ds_opt_name)
            if not opt_ds.data.records:
                raise ValueError(f"Dataset '{ds_opt_name}' exists but contains no entries.")
        except Exception as e:
            logger.info(f"Warning: Error accessing dataset '{ds_opt_name}': {e}, might not exist. Will continue without")
            continue

        # Get entries and their status
        incomplete_entries = get_incomplete_entries(opt_ds, opt_lot)

        # Handle incomplete entries
        while incomplete_entries:
            logger.info(f"Dataset '{ds_opt_name}' has {len(incomplete_entries)} incomplete entries. Waiting for completion...")
            time.sleep(60) 

            # Refresh the dataset to get the latest status
            opt_ds = client.get_collection("OptimizationDataset", ds_opt_name)
            incomplete_entries = get_incomplete_entries(opt_ds, opt_lot)

        logger.info(f"All entries in dataset '{ds_opt_name}' are complete.")
        processed_datasets.append(opt_ds)

    logger.info(f"Finished optimization datasets: {[d.name for d in processed_datasets]}\n\n\n")
    return processed_datasets

def process_be_computation(
    client: FractalClient,
    logger: logging.Logger,
    finished_opt_list: list,
    surf_opt_ds: ReactionDataset,
    smol_mol: ptl.Molecule,
    opt_lot: str,
    opt_method: str,
    mult: int,
    args
) -> list:
    """
    Process the binding energy computation for the given optimization datasets.

    Parameters:
    - client (FractalClient): A connection to the QCFractal server.
    - logger (logging.Logger): The logger instance for logging messages.
    - finished_opt_list (list): A list of optimization datasets that are complete.
    - surf_opt_ds (ReactionDataset): The surface optimization dataset.
    - smol_mol (ptl.Molecule): The small molecule for the reaction.
    - opt_lot (str): The level of theory, including method and basis.
    - args: The command-line arguments.

    Returns:
    - list: A list of job IDs for submitted computations.
    """
    logger = logging.getLogger("beep")
    all_ids = []
    for ds_opt in finished_opt_list:
        padded_log(logger, f"Checking {ds_opt.name} for repeated structures", padding_char="*", total_length=60)
        opt_stru = rmsd_filter(ds_opt, opt_lot, logger)  # Pass logger to rmsd_filter

        # Create or get benchmark binding energy dataset
        padded_log(logger, f"Building name for the new ReactionDataset", padding_char="*", total_length=60)
        cluster_name = "_".join(list(opt_stru.keys())[0].split("_")[1:3])
        cluster_mol = surf_opt_ds.get_record(cluster_name, opt_lot).get_final_molecule()
        rdset_name = f"be_{args.molecule.upper()}_{cluster_name.upper()}_{opt_method.upper()}"
        logger.info(f"ReactionDataset name for {ds_opt.name} is: {rdset_name}")

        padded_log(logger, f"Creating the dataset {rdset_name}", padding_char="*", total_length=60)
        ds_be = create_or_load_reaction_dataset(client, rdset_name, opt_lot, smol_mol, cluster_mol, ds_opt, opt_stru, logger)  # Pass logger

        keyword = None

        # Add uks keyword for open shell species.
        if mult == 2:
            keyword_obj = ptl.models.KeywordSet(values={"reference": "uks"})
            keyword = "rad_be"
            try:
                ds_be.add_keywords(keyword, args.program, keyword_obj, default=True)
                ds_be.save()
            except KeyError:
                pass

        padded_log(logger, f"Sending computations for {rdset_name}", padding_char="*", total_length=60)
        job_ids = compute_be_dft_energies(ds_be, args.level_of_theory, args.energy_tag, program=args.program, keyword=keyword, logger=logger)  # Pass logger
        all_ids.extend(job_ids)
        logger.info(f"Finished processing {rdset_name}\n\n\n")

    return all_ids


def main() -> None:
    args = get_arguments()
    logger = setup_logging("be_beep_log", args.molecule)
    logger.info(welcome_msg)

    client = ptl.FractalClient(
        address=args.client_address,
        verify=False,
        username=args.username,
        password=args.password
    )

    padded_log(logger, f"Checking for the state of the OptimizationDatasets")
    opt_lot = args.opt_level_of_theory
    surf_opt_ds, smol_mol = check_collections(client, args.surface_model_collection, args.small_molecule_collection, args.molecule, opt_lot)

    # Check multiplicity
    mult = smol_mol.molecular_multiplicity
    logger.info(f"\nThe molecular multiplicity is {mult}\n")

    opt_method = opt_lot.split("_")[0]
    opt_basis = opt_lot.split("_")[1]

    # Process binding energy computations
    if args.level_of_theory:
        padded_log(logger, f"Starting computation of Binding energies")
        be_parameters = (
            f"Binding Energy Computation Parameters:\n"
            f"- Molecule: {args.molecule}\n"
            f"- Surface Model Collection: {args.surface_model_collection}\n"
            f"- Small Molecule Collection: {args.small_molecule_collection}\n"
            f"- Level of Theory: {' '.join(args.level_of_theory)}\n"
            f"- Optimization Level of Theory: {opt_lot}\n"
            f"- Program: {args.program}\n"
            f"- Energy Tag: {args.energy_tag or 'Default: energies'}\n"
        )
        logger.info(be_parameters)
        padded_log(logger, f"Extracting OptimizationDatasets with no INCOMPLETE optimization")
        finished_opt_list = get_optdataset(client, surf_opt_ds, args.molecule, opt_lot, args.exclude_clusters)
        padded_log(logger, f"Creating {len(finished_opt_list)} ReactionDatasets for BE computation:")
        all_ids = process_be_computation(client, logger, finished_opt_list, surf_opt_ds, smol_mol, opt_lot, opt_method, mult, args)
        padded_log(logger, f"Checking for completion of ALL binding energy computations")
        check_jobs_status(client, all_ids, logger, wait_interval=600) 

    # Process Hessian computations
    if args.hessian_clusters:
        hessian_parameters = (
            f"Hessian Computation Parameters:\n"
            f"- Molecule: {args.molecule}\n"
            f"- Hessian Clusters: {', '.join(args.hessian_clusters)}\n"
            f"- Optimization Level of Theory: {opt_lot}\n"
            f"- Program: {args.program}\n"
            f"- Hessian Tag: {args.hessian_tag or 'Default: hessian'}\n"
        )
        padded_log(logger, f"Starting computation of Hessians")
        logger.info(hessian_parameters)
        all_hess_ids = []
        for cluster_name in args.hessian_clusters:
            rdset_name = f"be_{args.molecule.upper()}_{cluster_name.upper()}_{opt_method.upper()}"
            hess_ids = compute_hessian(client, rdset_name, opt_lot, mult, args.hessian_tag, logger=logger, program=args.program)  # Pass logger
            all_hess_ids.extend(hess_ids)
        padded_log(logger, f"Checking for completion of ALL Hessian computations")
        check_jobs_status(client, all_hess_ids, logger, wait_interval=30, print_job_ids = True)  # Pass logger

    logger.info("\nThank you for using the binding energy and hessian compute suite!")


if __name__ == "__main__":
    main()

