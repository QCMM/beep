# Standard library imports
import argparse
import functools
import logging
import os
import pickle
import sys
import time
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import qcfractal.interface as ptl
from collections import Counter
from pathlib import Path
from qcfractal.interface.client import FractalClient
from qcfractal.interface.collections import Dataset, OptimizationDataset, ReactionDataset
from qcelemental.models.molecule import Molecule
from typing import Any, Dict, List, Tuple, Union, NoReturn

# Local application/library specific imports
from beep.utils.dft_functionals import *
from beep.utils.logging_utils import *
from beep.utils.plotting_utils import *

# Configurations
warnings.filterwarnings("ignore")

bcheck = "\u2714"
mia0911 = "\u2606"
gear = "\u2699"

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
Welcome to the BEEP Binding Energy Evaluation Platform Geometry Benchmark Suite
---------------------------------------------------------------------------------------

The BEEP geometry benchmark suite  stands as a beacon  of validation for DFT geometries of 
binding sites. With steadfast accuracy and rigor, it  reports the mean RMSD for each 
structure. It offers plots and histograms  for nuanced comparison, and in its final act, 
adjudicates the most  commendable DFT method  for each family of  XC functionals – a 
testament to our  unwavering quest  for precision.  For  referenece, we stand on the 
shoulders of giants  CCSD(T)/CBS//CCSD(T)-F12/cc-pVDZ, the pillars of reliability, guide 
our relentless pursuit of truth.

Embark now on a journey of discovery, where data shepherds us to excellence—the very 
zenith of our scientific aspiration. 

Shine, Loom, Manifest.

                            By:  Stefan Vogt-Geisse
"""


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the script.

    Returns:
    - Namespace containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="""
A command line interface to sample the surface of a set of water clusters (stored in a 
QCFractal DataSet) with a small molecule or atom. This CLI is part
of the Binding Energy Evaluation Platform (BEEP).
    """
    )
    parser.add_argument(
        "--client_address",
        default="localhost:7777",
        help="The URL address and port of the QCFractal server (default: localhost:7777)",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="The username for the database client (Default = None)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="The password for the database client (Default = None)",
    )
    parser.add_argument(
        "--benchmark-structures",
        type=str,
        nargs="+",
        help="The name of the structures to use for the benchmark",
    )
    parser.add_argument(
        "--small-molecule-collection",
        default="Small_molecules",
        help="The name of the collection containing molecules or radicals (default: Small_molecules)",
    )
    parser.add_argument(
        "--molecule",
        required=True,
        help="The name of the molecule to be sampled (from a QCFractal OptimizationDataSet collection)",
    )
    parser.add_argument(
        "--surface-model-collection",
        default="small water",
        help="The name of the collection with the set of water clusters (default: small_water)",
    )
    parser.add_argument(
        "--reference-geometry-level-of-theory",
        nargs=3,
        default=["df-ccsd(t)-f12", "cc-pvdz", "molpro"],
        help="The level of theory in the format: method basis program (default: df-ccsd(t)-f12 cc-pvdz molpro)",
    )
    parser.add_argument(
        "--tag-reference-geometry",
        help="The tag for the high level refrenece geometry optimization with a QCFractal manager",
    )
    parser.add_argument(
        "--dft-optimization-program",
        default='psi4',
        help="The program to use for the DFT geometry optimization (default: psi4)",
    )
    parser.add_argument(
        "--dft-optimization-keyword",
        default=None,
        type=int,
        help="Keywords id for the gradient computation  (default: None)",
    )
    parser.add_argument(
        "--tag-dft-geometry",
        help="The tag for the dft geometry optimization with a QCFractal manager",
    )
    return parser.parse_args()



def get_or_create_collection(
    client: FractalClient, dset_name: str, collection_type: str
) -> OptimizationDataset:
    """
    Get or create an optimization dataset collection.

    Args:
    - client: Fractal client instance to use.
    - dset_name: Name of the OptimizationDataset.

    Returns:
    - An instance of the OptimizationDataset.
    """
    try:
        ds_opt = client.get_collection(collection_type.__name__, dset_name)
        out_string = f"OptimizationDataset {dset_name} already exists, new sampled structures will be saved here."
    except KeyError:
        ds_opt = collection_type(dset_name, client=client)
        ds_opt.save()
        ds_opt = client.get_collection(collection_type.__name__, dset_name)
        out_string = f"Creating new OptimizationDataset: {dset_name}."
    return ds_opt


def create_benchmark_dataset_dict(benchmark_structs: list[str]) -> dict[str, str]:
    """
    Creates a dictionary of simplified dataset names from a list of benchmark structure names.

    This function processes each benchmark structure name in the provided list. It splits each name 
    into its constituent parts, assuming a structure of 'molecule_surface_other'. 
    It then creates a simplified name consisting of only the 'molecule' and 'surface' parts. 
    These simplified names are stored in a dictionary, with the original benchmark structure 
    names as keys.

    Parameters:
    benchmark_structs (list[str]): A list of benchmark structure names, each expected to follow the format 'molecule_surface_other'.

    Returns:
    dict[str, str]: A dictionary where each key is the original benchmark structure name and each value is the simplified 'molecule_surface' name.
    """
    dataset_dict = {}
    for bchmk_struc_name in benchmark_structs:
        mol, surf, _ = bchmk_struc_name.split("_")
        dataset_dict[bchmk_struc_name] = f"{mol}_{surf}"
    return dataset_dict


def check_collection_existence(
    client: FractalClient,
    *collections: List,
    collection_type: str = "OptimizationDataset",
) -> None:
    """
    Check the existence of collections and raise DatasetNotFound error if not found.

    Args:
    - client: QCFractal client object
    - *collections: List of QCFractal Datasets.
    - collection_type: type of Optimization Dataset

    Raises:
    - DatasetNotFound: If any of the specified collections do not exist.
    """
    logger = logging.getLogger("beep")
    for collection in collections:
        try:
            client.get_collection(collection_type, collection)
        except KeyError:
            raise DatasetNotFound(
                f"Collection {collection} does not exist. Please create it first. Exiting..."
            )
        logger.info(f"The {collection_type} named {collection} exsits {bcheck}")


def create_and_add_specification(
    client: FractalClient,
    odset: OptimizationDataset,
    method: str,
    basis: str,
    program: str,
    qc_keyword: int,
    geom_keywords: int = None,
):
    """
    Create and add a specification to an optimization dataset.

    Args:
    - odset (OptimizationDataset): The dataset to which the specification is added.
    - method (str): The computational method.
    - basis (str): The basis set.
    - program (str): The program used for the computation.
    - geom_keywords (int, optional): Keyword ID for the geometric optimization. Default is None.
    - qc_keyword (int, optional): Keyword ID for the quantum chemistry computation. Default is None.

    Returns:
    - str: The name of the created specification.
    """
    logger = logging.getLogger("beep")
    spec_name = f"{method}_{basis}"
    if qc_keyword:
        kw_name = client.query_keywords()[qc_keyword].values.values()
        logger.debug(f"Using the following keyword for the specification {kw_name} to {odset.name}")
        if ("uks" or "uhf") in kw_name and program == "psi4":
            spec_name = "U"+spec_name
    print(spec_name)
    spec = {
        "name": spec_name,
        "description": f"Geometric {program}/{method}/{basis}",
        "optimization_spec": {"program": "geometric", "keywords": geom_keywords},
        "qc_spec": {
            "driver": "gradient",
            "method": method,
            "basis": basis,
            "keywords": qc_keyword,
            "program": program,
        },
    }
    odset.add_specification(**spec, overwrite=True)
    odset.save()
    logger.debug(f"Create and added the specifiction {spec_name} to {odset.name}")
    return spec_name


def get_molecular_multiplicity(
    client: FractalClient, dataset: OptimizationDataset, molecule_name: str
) -> int:
    """
    Get the molecular multiplicity of a molecule in an optimization dataset.

    Args:
    - client (FractalClient): The client connected to a QCFractal server.
    - dataset_name (OptimizationDataset): The name of the optimization dataset.
    - molecule_name (str): The name of the molecule.

    Returns:
    - int: The molecular multiplicity of the specified molecule.
    """
    initial_molecule_id = dataset.data.records[molecule_name.lower()].initial_molecule
    mol = client.query_molecules(initial_molecule_id)[0]
    return mol.molecular_multiplicity


def optimize_reference_molecule(
    odset: OptimizationDataset, struct_name: str, geom_ref_opt_lot: str, mol_mult: int, opt_tag: str,
) -> None:
    """
    Optimize a molecule in the dataset based on its molecular multiplicity.

    Args:
    - odset (OptimizationDataset): The dataset containing the molecule.
    - struct_name (str): The name of the structure to optimize.
    - geom_ref_opt_lot (str): The level of theory for optimization.
    - mol_mult (int): The molecular multiplicity.

    Returns:
    - int: The number of computations submitted.
    """
    if (mol_mult) == 1 or (mol_mult == 2):
        return odset.compute(geom_ref_opt_lot, tag=opt_tag, subset={struct_name})
    else:
        raise RuntimeError(
            "Invalid value for molecular multiplicity. It has to be 1 (Singlet) or 2 (Doublet)"
        )


def optimize_dft_molecule(
    client: FractalClient, odset: OptimizationDataset, struct_name: str, method: str, basis: str, program: str, dft_keyword: int, opt_tag: str
) -> int:
    """
    Create and submit a computation job for a given structure with specified method and basis.

    Args:
    - odset (OptimizationDataset): The dataset to which the job is submitted.
    - struct_name (str): Name of the structure for optimization.
    - method (str): Computational method.
    - basis (str): Basis set.
    - program (str): Program to use for computation.

    Returns:
    - int: The number of jobs submitted.
    """
    logger = logging.getLogger("beep")
    spec_name = create_and_add_specification(client, odset, method, basis, program, dft_keyword)
    cr = odset.compute(spec_name, tag=opt_tag, subset={struct_name})
    return cr


def wait_for_completion(
    client: FractalClient,
    odset_dict: Dict[str, "OptimizationDataset"],
    opt_lot: Union[str, List[str]],
    program: str,
    qc_keyword: int =  None,
    wait_interval: int = 600,
    check_errors: bool = False,
) -> int:
    """
    Waits indefinitely for all entries in the given OptimizationDatasets to complete, with an optional check for errors.
    Returns a summary of job statuses upon completion. Can handle a single optimization lot or a list of them.

    Args:
        odset_dict (Dict[str, "OptimizationDataset"]): Dictionary with structure names as keys and OptimizationDatasets as values.
        opt_lot (Union[str, List[str]]): The specification (level of theory) to check. Can be a string or a list of strings.
        wait_interval (int): Interval in seconds between status checks.
        dft_keyword
        check_errors (bool): If True, raise an error if any entry has a status of 'ERROR'.

    Raises:
        RuntimeError: If check_errors is True and any entry has a status of 'ERROR'.

    Returns:
        str: Summary message indicating the count of each job status.
    """
    logger = logging.getLogger("beep")
    if isinstance(opt_lot, str):
        opt_lot = [opt_lot]

    logger.info("\nChecking if the computations have finished") # for the following levels of theory:")
    logger.info("\n")
    while True:
        statuses = []
        for lot in opt_lot:
            try:
                if ("uks" or "uhf") in client.query_keywords()[qc_keyword].values.values() and program == "psi4":
                    lot = "U"+lot
            except TypeError:
                pass
            for struct_name, odset in odset_dict.items():
                status = odset.get_record(struct_name, specification=lot).status
                statuses.append(status)

                if status == "ERROR" and check_errors:
                    raise RuntimeError(
                        f"Error encountered in computation for {struct_name} with spec '{lot}'"
                    )

        status_counts = Counter(statuses)

        if status_counts["INCOMPLETE"] == 0:
            status_message = ", ".join(
                [f"{status}: {count}" for status, count in status_counts.items()]
            )
            logger.info(f"All entries have been processed. (Complete: {status_counts['COMPLETE']}, ERROR: {status_counts['ERROR']}) {bcheck}")
            return status_counts['COMPLETE']

        logger.info(
            f"Waiting for {wait_interval} seconds before rechecking statuses... (Incomplete: {status_counts['INCOMPLETE']})"
        )
        time.sleep(wait_interval)


def check_dataset_status(
    client: FractalClient,
    dataset: ReactionDataset,
    cbs_list: List[str],
    wait_interval: int = 600,
) -> None:
    """
    Continuously monitors and reports the status of computations in a dataset.

    This function checks the status of computational records in the given ReactionDataset 
    for various methods and bases specified in `cbs_list`. It categorizes the status of each 
    record as COMPLETE, INCOMPLETE, or ERROR. The function keeps running until all records are 
    complete or if there's an error in any record, and it prints the status summary for each 
    method every 600 seconds.

    Parameters:
    client (Any): The client object used to interact with the dataset.
    dataset (Any): The dataset object containing computational records.
    cbs_list (List[str]): A list of strings representing combinations of computational methods and basis sets.

    Returns:
    None: This function returns None but will print status updates and may raise an exception if an error is detected in the dataset records.
    """
    while True:
        status_counts = {
            method: {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}
            for method in set(lot.split("_")[0] for lot in cbs_list)
        }

        for lot in cbs_list:
            method, basis = lot.split("_")
            if "ccsd" in method:
                df = dataset.get_records(
                    method=method, basis=basis, program="psi4", keywords="df"
                )
            else:
                df = dataset.get_records(method=method, basis=basis, program="psi4")

            # Count the statuses for the current method
            for index, row in df.iterrows():
                status = row["record"].status.upper()
                if status not in status_counts[method]:
                    continue  # If status is not one of the expected statuses, skip it

                status_counts[method][status] += 1

                if status == "ERROR":
                    raise Exception(
                        f"Error in record {index} with level of theory {lot}"
                    )

        # Print the summary counts for each method
        for method, counts in status_counts.items():
            print(
                f"{method}: {counts['INCOMPLETE']} INCOMPLETE, {counts['COMPLETE']} COMPLETE, {counts['ERROR']} ERROR"
            )

        # Check if all records are complete or any record has an error
        all_complete = all(
            counts["INCOMPLETE"] == 0 and counts["ERROR"] == 0
            for counts in status_counts.values()
        )
        if all_complete:
            print("All records are COMPLETE.")
            break
        if any(counts["ERROR"] > 0 for counts in status_counts.values()):
            print("There are records with ERROR.")
            break

        time.sleep(600)  # Wait for 600 seconds before the next check

def compute_rmsd(
    mol1: Molecule, 
    mol2: Molecule, 
    rmsd_symm: bool
) -> Tuple[float, float]:
    """
    Computes the root-mean-square deviation (RMSD) between two molecular structures.

    This function calculates the RMSD between two Molecule objects. If rmsd_symm is True, 
    it also calculates the RMSD considering the mirror image of the first molecule. It returns a 
    tuple containing the RMSD value and the mirrored RMSD value (which defaults to 10.0 
    if rmsd_symm is False).

    Parameters:
    mol1 (Molecule): The first molecule object.
    mol2 (Molecule): The second molecule object to which the first is compared.
    rmsd_symm (bool): A boolean indicating whether to compute the RMSD for the mirrored structure 
    of mol1 as well.

    Returns:
    Tuple[float, float]: A tuple containing the RMSD value and the mirrored RMSD value.
    """
    rmsd_val_mirror = 10.0
    if rmsd_symm:
        align_mols_mirror = mol1.align(mol2, run_mirror=True)
        rmsd_val_mirror = align_mols_mirror[1]["rmsd"]
    align_mols = mol1.align(mol2, atoms_map=True)
    rmsd_val = align_mols[1]["rmsd"]

    if rmsd_val < rmsd_val_mirror:
        return rmsd_val
    else:
        return rmsd_val_mirror

    return rmsd_val, rmsd_val_mirror


def compare_rmsd(
    dft_lot: List[str],
    odset_dict: Dict[str, OptimizationDataset],
    ref_geom_fmols: Dict[str, Molecule]
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    """
    Compares RMSD values for different levels of theory in a DFT optimization lot.

    This function calculates the RMSD and mirrored RMSD of final molecules in an 
    optimization dataset against reference geometries. It then averages these values 
    for each level of theory (opt_lot) in the DFT optimization lot and determines the 
    level of theory that yields the lowest average RMSD.

    Parameters:
    dft_lot (List[str]): A list of strings representing different levels of theory.
    odset_dict (Dict[str, Any]): A dictionary mapping structure names to optimization dataset objects.
    ref_geom_fmols (Dict[str, Molecule]): A dictionary mapping structure names to reference geometry molecules.

    Returns:
    Tuple[Dict[str, float], Dict[str, float]]: A tuple containing two dictionaries. The first dictionary maps 
    the best level of theory to its average RMSD value of the different groups. The second dictionary maps all 
    levels of theory to their respective average RMSD values.
    """
    logger = logging.getLogger("beep")
    logger.propagate = False
    rmsd_df = pd.DataFrame(index=odset_dict.keys(), columns=dft_lot)
    final_opt_lot = {}
    total_operations = len(dft_lot)

    for i, opt_lot in enumerate(dft_lot):
        rmsd_tot_dict = {}

        for struct_name, odset in odset_dict.items():
            record = odset.get_record(struct_name, specification=opt_lot)
            err = record.get_error()
            if err:
                logger.info(f"Calculation for {struct_name} at the {opt_lot} level of theory finished with error. It will be skipped")
                break
            fmol = record.get_final_molecule()
            rmsd = compute_rmsd(ref_geom_fmols[struct_name], fmol, rmsd_symm=True)
            rmsd_tot_dict[struct_name] = rmsd
            rmsd_df.at[struct_name, opt_lot] = rmsd

        if err:
           continue
        rmsd_tot = list(rmsd_tot_dict.values())
        final_opt_lot[opt_lot] = np.mean(rmsd_tot)
        log_progress(logger, i + 1, total_operations)

    # Find the lowest average RMSD value
    lowest_values = sorted(final_opt_lot.values())[:1]
    best_geom_lot = {k: v for k, v in final_opt_lot.items() if v in lowest_values}

    return best_geom_lot, final_opt_lot, rmsd_df

def compare_all_rmsd(
    functional_groups: Dict[str, List[str]], 
    odset_dict: Dict[str, OptimizationDataset], 
    ref_geom_fmols: Dict[str, Molecule]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compares RMSD values across different functional groups and their respective functionals.

    This function iterates over various functional groups, each with a list of functionals. 
    For each functional group, it calls the 'compare_rmsd' function to determine the best 
    optimization level of theory and gathers all final optimization levels of theory. 
    It returns a dictionary mapping each functional group to its best optimization level 
    and a dictionary with all final optimization levels of theory.

    Parameters:
    functional_groups (Dict[str, List[str]]): A dictionary mapping functional group names 
    to lists of strings representing different levels of theory.
    odset_dict (Dict[str, Any]): A dictionary mapping structure names to optimization dataset objects.
    ref_geom_fmols (Dict[str, Molecule]): A dictionary mapping structure names to reference geometry molecules.

    Returns:
    Tuple[Dict[str, Dict[str, float]], Dict[str, float]]: A tuple containing two dictionaries. The first maps each functional group 
    to its best optimization level of theory. The second contains all final optimization levels of theory for each functional group.
    """
    logger = logging.getLogger("beep")
    best_opt_lot = {}

    combined_rmsd_df = pd.DataFrame()

    for func_group, functionals in functional_groups.items():
        logger.info(f"\nProcessing RMSD for {func_group} type methods:")
        group_best_opt_lot, final_opt_lot, rmsd_df = compare_rmsd(
            functionals, odset_dict, ref_geom_fmols
        )

        # Rename columns to include the functional group name
        rmsd_df.columns = [f"{func_group}_{col}" for col in rmsd_df.columns]

        # Join the DataFrame with the combined DataFrame
        combined_rmsd_df = pd.concat([combined_rmsd_df, rmsd_df], axis=1)

        best_opt_lot[func_group] = group_best_opt_lot

    return best_opt_lot, combined_rmsd_df



def save_df_to_json(logger, df, filename):
    """
    Save a pandas DataFrame to a JSON file.

    Parameters:
    logger
    df (pandas.DataFrame): The DataFrame to save.
    filename (str): The name of the file where the DataFrame will be saved.
    """
    logger = logging.getLogger("beep")
    try:
        # df.to_json(filename, orient='records', lines=True)
        df.to_json(filename)
        logger.info(f"\nDataFrame successfully saved to {filename}\n")
    except Exception as e:
        logger.info(f"Error saving DataFrame to JSON: {e}")


def main():
    # Call the arguments
    args = parse_arguments()


    logger = setup_logging("bchmk_geom",args.molecule)
    logger.info(welcome_msg)

    client = ptl.FractalClient(
        address=args.client_address,
        verify=False,
        username=args.username,
        password=args.password,
    )

    # Tags for the optimization 
    hl_tag  = args.tag_reference_geometry
    dft_tag = args.tag_dft_geometry

    # The name of the molecule to be sampled at level of theory opt_lot
    smol_name = args.molecule
    gr_method, gr_basis, gr_program = args.reference_geometry_level_of_theory
    geom_ref_opt_lot = gr_method + "_" + gr_basis

    bchmk_structs = args.benchmark_structures
    surf_dset_name = args.surface_model_collection
    smol_dset_name = args.small_molecule_collection
    padded_log(logger, "Starting BEEP geometry benchmark procedure", padding_char=gear)
    logger.info(f"Molecule: {smol_name}")
    logger.info(f"Surface Model: {smol_dset_name}")
    logger.info(f"Benchmark Structures: {bchmk_structs}")
    
    # Defining lists and Dictionaries
    w_list = []
    odset_dict = {}
    bchmk_dset_names = {}

    # Save name of collections with the Benchmark strcutrues in a dictionary
    bchmk_dset_names = create_benchmark_dataset_dict(bchmk_structs)

    # Check existance of the collections
    check_collection_existence(client, *bchmk_dset_names.values())
    check_collection_existence(client, smol_dset_name)
    check_collection_existence(client, surf_dset_name)

    # Get multiplcity
    smol_dset = client.get_collection("OptimizationDataset", smol_dset_name)
    mol_mult = get_molecular_multiplicity(client, smol_dset, smol_name)
    logger.info(f"\n The molecular multiplicity of {smol_name} is {mol_mult}")

    # Populate the odset_dict with different collections
    odset_dict = {smol_name: smol_dset}

    for bchmk_struct_name, odset_name in bchmk_dset_names.items():
        odset_dict[bchmk_struct_name] = client.get_collection(
            "OptimizationDataset", odset_name
        )
        surf_mod = bchmk_struct_name.split("_")[1].upper()
        odset_dict[surf_mod] = client.get_collection(
            "OptimizationDataset", surf_dset_name
        )

    # Specification for reference of geometry benchmark:

    padded_log(logger, "Start of the geometry refrence processing")
    logger.info(f"Method: {gr_method}")
    logger.info(f"Basis: {gr_basis}")
    logger.info(f"Program: {gr_program}\n")
    for odset in odset_dict.values():
        create_and_add_specification(
            client,
            odset,
            method=gr_method,
            basis=gr_basis,
            program=gr_program,
            qc_keyword=None,
            geom_keywords=None,
        )

    # Optimize all three molecules at the reference benchmark level of theory
    ct = 0
    for struct_name, odset in odset_dict.items():
        ct += optimize_reference_molecule(
            odset, struct_name, geom_ref_opt_lot, mol_mult, hl_tag
        )

    logger.info(f"\nSend a total of {ct} structures to compute at the {geom_ref_opt_lot} level of theory to the tag {hl_tag}\n")

    ## Wait for the Opimizations to finish
    wait_for_completion(client, odset_dict, geom_ref_opt_lot, gr_program, wait_interval=600, check_errors=True)

    padded_log(logger, "Start of the DFT geometry computations")

    ## Optimize with DFT functionals
    dft_program = args.dft_optimization_program
    dft_keyword = args.dft_optimization_keyword
    print("THE DFT KEYWORD: ", dft_keyword)

    ## Saving funcitona lists in a dictionary
    dft_geom_functionals = {
        "geom_hmgga_dz": geom_hmgga_dz(),
        "geom_hmgga_tz": geom_hmgga_tz(),
        "geom_gga_dz": geom_gga_dz(),
        "geom_sqm_mb": geom_sqm_mb(),
    }

    # Combine all functionals into one list
    all_dft_functionals = [
        functional
        for functionals in dft_geom_functionals.values()
        for functional in functionals
    ]
   
    logger.info(f"Program: {dft_program}")
    logger.info(f"DFT and SQM geometry methods:")
    dict_to_log(logger, dft_geom_functionals)

    # Sending all DFT geometry optimizations
    ct = 0
    c = 0
    padded_log(logger,'Start sending DFT optimizations')
    for struct_name, odset in odset_dict.items():
        logger.info(f"\nSending geometry optimizations for {struct_name}")
        cs = 0
        for functionals in dft_geom_functionals.values():
            for functional in functionals:
                method, basis = functional.split("_")
                cs += optimize_dft_molecule(client, odset, struct_name, method, basis, dft_program, dft_keyword, dft_tag)
                ct += cs
                c+=1
        logger.info(f"Send {cs} geometry optimizations for structure {struct_name}")

    logger.info(f"\nSend {c}/{ct} to the tag {dft_tag}")

    wait_for_completion(
        client, odset_dict, all_dft_functionals, dft_program, dft_keyword, wait_interval=200, check_errors=False
    )

    # Save optimized molecules of the reference structures
    ref_geom_fmols = {}
    for struct_name, odset in odset_dict.items():
        record = odset.get_record(struct_name, specification=geom_ref_opt_lot)
        ref_geom_fmols[struct_name] = record.get_final_molecule()

    padded_log(logger, "Start of RMSD comparsion between DFT and {} geometries", geom_ref_opt_lot)
    # Compare RMSD for all functional groups
    try:
        if ("uks" or "uhf") in client.query_keywords()[dft_keyword].values.values() and dft_program == "psi4":
            dft_geom_functionals = {key: ['U' + item for item in value] for key, value in dft_geom_functionals.items()}
    except TypeError:
        pass
    best_opt_lot, rmsd_df = compare_all_rmsd(
        dft_geom_functionals, odset_dict, ref_geom_fmols
    )
    # Logging the results of the geometry benchmark
    padded_log(logger, "BENCHMARK RESULSTS")
    log_dataframe_averages(logger, rmsd_df)

    # Define the folder path for 'json_data' in the current working directory
    folder_path_json = Path.cwd() / Path('geom_json_data_' + smol_name)
    if not folder_path_json.is_dir():
        folder_path_json.mkdir(parents=True, exist_ok=True)

    # Save json with all the results
    save_df_to_json(logger, rmsd_df, str(folder_path_json)+"/results_geom_benchmark.json")

    folder_path_plots = Path.cwd() / Path('geom_bchmk_plots_' + smol_name)
    if not folder_path_plots.is_dir():
        folder_path_plots.mkdir(parents=True, exist_ok=True)

    rmsd_histograms(rmsd_df, smol_name, str(folder_path_plots))

    padded_log(logger, "Geometry Benchmark finished successfully! Hasta pronto!", padding_char=mia0911)

    # END GEOMETRY BENCHMARK, INCLUDE SOME RMSD PLOTS IN THE FUTURE.
    return None

if __name__ == "__main__":
    main()
