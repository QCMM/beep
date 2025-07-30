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
import qcelemental as qcel
from collections import Counter
from pathlib import Path
from qcfractal.interface.client import FractalClient
from qcfractal.interface.collections import (
    Dataset,
    OptimizationDataset,
    ReactionDataset,
)
from qcelemental.models.molecule import Molecule
from typing import Any, Dict, List, Tuple, Union, NoReturn

# Local application/library specific imports
from beep.utils.dft_functionals import *
from beep.utils.logging_utils import *
from beep.utils.plotting_utils import *
from beep.utils.cbs_extrapolation import *

bcheck = "\u2714"
mia0911 = "\u2606"
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
Welcome to the BEEP Binding Energy Evaluation Platform  Binding Energy Benchmark Suite
---------------------------------------------------------------------------------------

In a quest to forge the path to rigor, the BEEP binding energy benchmark workflow establishes 
itself as a cornerstone of validation for DFT binding energies of small molecules bound 
to cluster models. We harness from the monumental gold standart  CCSD(T)/CBS//CCSD(T)-F12/cc-pVDZ-F12, 
the stalwarts of reliability, guiding us in our relentless pursuit to accuracy.
With unwavering  diligence, it reports the mean absolute error for each functional, providing 
detailed plots  and histograms for comprehensive comparison. In its decisive act, 
echoing a firm resolution,  it identifies the most commendable DFT method for each 
family of XC functionals. Furthermore, the workflow tests the sensitivity of the underlying 
binding site geometry,  allowing for the examination of all DFT functionals across various 
geometry qualities. 


Steadfastness, Learning, and Mastery.

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
        help="The name of the molecule to be sampled (entry from a QCFractal OptimizationDataSet collection)",
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
        "--optimization-level-of-theory",
        type=str,
        nargs="+",
        help="The optimiziation levels of theory for which the DFT energies should be computed in the format method_basis (e.g. M05_def2-svp)",
    )
    # parser.add_argument(
    #    "--reference-energy-level-of-theory",
    #    nargs=3,
    #    default=["ccsd(t)", "cbs", "psi4"],
    #    help="The level of theory and program for the geometry refinement in the format: method_basis (default: df-ccsd(t) cc-pvdz molpro)",
    # )
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
    logger = logging.getLogger("beep")
    try:
        ds = client.get_collection(collection_type.__name__, dset_name)
        logger.info(
            f"Collection of type {collection_type.__name__} with name {dset_name} already exists. {bcheck}\n"
        )
    except KeyError:
        ds = collection_type(dset_name, client=client)
        ds.save()
        ds = client.get_collection(collection_type.__name__, dset_name)
        logger.info(f"Creating new {collection_type.__name__}: {dset_name}.\n")
    return ds


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
        logger.info(f"The {collection_type} named {collection} exsits {bcheck}\n")


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


def populate_dataset_with_structures(
    cbs_col: Dataset,
    ref_geom_fmols: Dict[str, Molecule],
    bchmk_structs: List[str],
    odset_dict: Dict[str, OptimizationDataset],
    geom_ref_opt_lot: str,
) -> None:
    """
    Populates a dataset with molecular structures, including fragments if applicable.

    Parameters:
    - cbs_col: The collection object representing the dataset in which entries are added.
    - ref_geom_fmols: A dictionary of reference optimized geometries for molecules.
    - bchmk_structs: A list of benchmark structures to be included.
    - odset_dict: A dictionary mapping names to their respective dataset objects.
    - geom_ref_opt_lot: The level of theory used for the reference geometry optimization.

    This function adds each molecular structure from `ref_geom_fmols` to `cbs_col`. If the structure
    is listed in `bchmk_structs`, it also computes and adds its molecular fragments.
    """

    logger = logging.getLogger("beep")
    for name, fmol in ref_geom_fmols.items():
        if name in bchmk_structs:
            mol_name, surf_name, _ = name.split("_")
            surf_mod_mol = (
                odset_dict[surf_name.upper()]
                .get_record(name=surf_name.upper(), specification="ccsd(t)_cc-pvtz")
                #.get_record(name=surf_name.upper(), specification=geom_ref_opt_lot)
                .get_final_molecule()
            )
            len_f1 = len(surf_mod_mol.symbols)
            mol_f1, mol_f2 = create_molecular_fragments(fmol, len_f1)
            cbs_col.add_entry(name, fmol)
            cbs_col.add_entry(name + "_f1", mol_f1)
            cbs_col.add_entry(name + "_f2", mol_f2)
            logger.info(f"Adding molecule and fragments of {name} to {cbs_col.name}")

            cbs_col.save()
        else:
            cbs_col.add_entry(name, fmol)
            logger.info(f"Adding molecule {name} to {cbs_col.name}")

            cbs_col.save()


def add_cc_keywords(cbs_col: Dataset, mol_mult: int) -> None:
    """
    Adds keywords for coupled cluster calculations to the collection based on the multiplicity of the molecule.

    Parameters:
    - cbs_col: The Dataset collection to which the keywords will be added.
    - mol_mult (int): The multiplicity of the molecule.

    Depending on the multiplicity of the molecule, specific keywords are added to the collection for
    density-fitted or unrestricted Hartree-Fock coupled cluster calculations.
    """
    logger = logging.getLogger("beep")
    if mol_mult == 1:
        logger.info(
            f"\n\nCreating keywords for closed shell coupled cluster computation"
        )
        kw_dict = {"scf_type": "df", "cc_type": "df", "freeze_core": "true"}
        logger.info(f"Keywords dictionary: {kw_dict}")
        kw_dfit = ptl.models.KeywordSet(values=kw_dict)
    elif mol_mult == 2:
        logger.info(f"\n\nCreating keywords for open shell coupled cluster computation")
        kw_dict = {"reference": "uhf", "freeze_core": "true"}
        logger.info(f"Keywords dictionary: {kw_dict}")
        kw_dfit = ptl.models.KeywordSet(values=kw_dict)

    try:
        cbs_col.add_keywords("df", "psi4", kw_dfit)
        cbs_col.save()
        logger.info("Added keywords to Dataset\n")
    except KeyError:
        logger.info("Keyword already set in Dataset, nothing to add\n")


def compute_all_cbs(cbs_col: Dataset, cbs_list: List[str], mol_mult: int) -> List[str]:
    """
    Computes all Complete Basis Set (CBS) computations for the specified levels of theory.

    Parameters:
    - cbs_col: The Dataset collection object where the computations are to be added.
    - cbs_list (List[str]): A list of strings representing the levels of theory.
    - mol_mult (int): The multiplicity of the molecule.

    Returns:
    - List[str]: A list of IDs representing the computed CBS tasks.

    Depending on the molecule's multiplicity and the level of theory, the appropriate computations
    are added to the collection. The function returns a list of IDs for these computations.
    """

    all_cbs_ids = []
    logger = logging.getLogger("beep")
    id_str = ""
    dir_path = Path("cbs_ids")

    # Check if the directory exists, if not create it
    if not dir_path.exists():
        dir_path.mkdir()
    file_path = dir_path / "cbs_ids.dat"

    for lot in cbs_list:
        # Splitting the level of theory into method and basis
        method, basis = lot.split("_")[0], lot.split("_")[1]
        logger.info(f"\nSendig computations for {method}/{basis}")

        # Tag assignment based on molecule multiplicity
        tag = "cbs_en_radical" if mol_mult == 2 else "cbs_en"

        # Check if "scf" is not in the level of theory
        if "scf" not in lot:
            c = cbs_col.compute(method, basis, tag=tag, keywords="df", program="psi4")
        else:
            c = cbs_col.compute(method, basis, tag=tag, program="psi4")

        id_li = c.submitted + c.existing
        id_str += f"{method}_{basis}: {id_li}\n"
        logger.info(f"Submited {len(c.submitted)} Computation to tag {tag}.")
        if len(c.existing) > 0:
            logger.info(f"{len(c.existing)} have already been computed {bcheck}")

    # Writing IDs to file for restart
    with file_path.open(mode="w") as file:
        file.write(id_str)

    return all_cbs_ids


def check_dataset_status(
    dataset: Dataset,
    cbs_list: List[str],
    wait_interval: int = 1800,
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
    logger = logging.getLogger("beep")
    while True:
        status_counts = {
            method: {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}
            for method in set(lot.split("_")[0] for lot in cbs_list)
        }

        for lot in cbs_list:
            method, basis = lot.split("_")
            if not "scf" in method:
                df = dataset.get_records(
                    method=method, basis=basis, program="psi4", keywords="df"
                )
            else:
                df = dataset.get_records(
                    method=method, basis=basis, program="psi4", keywords=None
                )

            # Count the statuses for the current method
            for index, row in df.iterrows():
                # print(index, row["record"])
                status = row["record"].status.upper()
                if status not in status_counts[method]:
                    continue  # If status is not one of the expected statuses, skip it

                status_counts[method][status] += 1

                if status == "ERROR":
                    raise Exception(
                        f"Error in record {index} with level of theory {lot}"
                    )

        # Print the summary counts for each method
        logger.info(f"\nChecking status of computations for CCSD(T)\CBS:\n")
        for method, counts in status_counts.items():
            logger.info(
                f"{method}: {counts['INCOMPLETE']} INCOMPLETE, {counts['COMPLETE']} COMPLETE, {counts['ERROR']} ERROR"
            )

        # Check if all records are complete or any record has an error
        all_complete = all(
            counts["INCOMPLETE"] == 0 and counts["ERROR"] == 0
            for counts in status_counts.values()
        )
        if all_complete:
            logger.info("\nAll records are COMPLETE. Continuing with the execution")
            break
        if any(counts["ERROR"] > 0 for counts in status_counts.values()):
            logger.info("\nThere are records with ERROR. Proceed with cuation")
            break

        time.sleep(wait_interval)  # Wait for 1800 seconds before the next check
    return None


def create_molecular_fragments(mol: Molecule, len_f1):
    geom = mol.geometry.flatten()
    symbols = mol.symbols
    f_mol = ptl.Molecule(
        symbols=symbols,
        geometry=geom,
        fragments=[
            list(range(0, len_f1)),
            list(range(len_f1, len(symbols))),
        ],
    )
    f1_mol = f_mol.get_fragment(0)
    f2_mol = f_mol.get_fragment(1)
    return f1_mol, f2_mol


def get_energy_record(ds: Dataset, struct: str, method: str, basis: str) -> Any:
    """
    Retrieves the energy record for a given structure using specified method and basis set.

    Parameters:
    ds (Dataset): The dataset object containing the energy records.
    struct (str): The name of the structure for which the energy record is required.
    method (str): The computational chemistry method (e.g., 'mp2', 'ccsd') used for the calculation.
    basis (str): The basis set used for the calculation.

    Returns:
    Any: The record object containing the energy information. The type of this object will depend on
         the implementation of the Dataset's `get_records` method.

    Note:
    - If the method is not 'scf', a 'df' keyword is added to the retrieval arguments.
    - The function assumes that 'ds.get_records' can handle the provided arguments and return the desired record.
    """
    kwargs = {"method": method, "basis": basis, "program": "Psi4", "keywords": None}
    if not "scf" in method:
        kwargs["keywords"] = "df"

    # ds.get_records(**kwargs).index = ds.get_records(**kwargs).index.str.upper()
    df_records = ds.get_records(**kwargs)
    df_records.index = df_records.index.str.upper()
    records = df_records.loc[struct.upper()]

    # print(ds.get_records(**kwargs).index )
    # records = ds.get_records(**kwargs).loc[struct.upper()]

    # Check if the result is a DataFrame (multiple records) or not (single record) and retrive the first record
    if isinstance(records, pd.DataFrame):
        rec = records.iloc[0].iloc[0]
    else:
        rec = records[0]
    return rec


def get_cbs_energy(ds: Dataset, struct: str, cbs_lot_list: List[str]) -> pd.DataFrame:
    """Calculates the CBS (Complete Basis Set) energy for a given structure based on
    a list of computational levels of theory.

    Parameters:
    ds (Dataset): The dataset object containing the energy records.
    struct (str): The name of the structure for which CBS energy is calculated.
    cbs_lot_list (List[str]): A list of computational levels of theory in the format 'method_basis'.

    Returns:
    float: The calculated CBS energy for the given structure.

    Note:
    - The function makes use of 'scf_xtpl_helgaker_3', 'scf_xtpl_helgaker_2', and 'corl_xtpl_helgaker_2'
      for extrapolation calculations. These functions should be defined and accessible in the current scope.
    - The energy calculations are based on extrapolation methods and might include multiple steps of calculation.
    - The function assumes 'ccsd' in the method requires special handling.
    """
    cbs_lot_en = {}
    # Initialize an empty DataFrame
    columns = ["SCF", "MP2", "CCSD", "CCSD(T)"]
    index = ["aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ", "CBS"]
    cbs_lot_df = pd.DataFrame(index=index, columns=columns)

    # Data extraction and computation
    for lot in cbs_lot_list:
        method, basis = lot.split("_")
        rec = get_energy_record(ds, struct, method, basis)

        if "mp2" in method:
            cbs_lot_df.at[basis, "MP2"] = rec.dict()["extras"]["qcvars"][
                "MP2 CORRELATION ENERGY"
            ]
        elif "ccsd(t)" in method:
            cbs_lot_df.at[basis, "MP2"] = rec.dict()["extras"]["qcvars"][
                "MP2 CORRELATION ENERGY"
            ]
            cbs_lot_df.at[basis, "CCSD"] = rec.dict()["extras"]["qcvars"][
                "CCSD CORRELATION ENERGY"
            ]
            cbs_lot_df.at[basis, "CCSD(T)"] = rec.dict()["extras"]["qcvars"][
                "CCSD(T) CORRELATION ENERGY"
            ]
        else:
            cbs_lot_df.at[basis, method.upper()] = rec.return_result

    # Update CCSD and CCSD(T) to get the delta correlation
    cbs_lot_df["CCSD(T)"] -= cbs_lot_df["CCSD"]
    cbs_lot_df["CCSD"] -= cbs_lot_df["MP2"]

    # Do CBS Extrapolations and contributions calculation
    cbs_lot_df.at["CBS", "SCF"] = scf_xtpl_helgaker_3(
        "scf_dtq_xtpl",
        2,
        cbs_lot_df.at["aug-cc-pVDZ", "SCF"],
        3,
        cbs_lot_df.at["aug-cc-pVTZ", "SCF"],
        4,
        cbs_lot_df.at["aug-cc-pVQZ", "SCF"],
    )

    cbs_lot_df.at["CBS", "MP2"] = corl_xtpl_helgaker_2(
        "mp2_tq",
        3,
        cbs_lot_df.at["aug-cc-pVTZ", "MP2"],
        4,
        cbs_lot_df.at["aug-cc-pVQZ", "MP2"],
    )

    cbs_lot_df.at["CBS", "CCSD"] = corl_xtpl_helgaker_2(
        "ccsd_dt",
        2,
        cbs_lot_df.at["aug-cc-pVDZ", "CCSD"],
        3,
        cbs_lot_df.at["aug-cc-pVTZ", "CCSD"],
    )

    cbs_lot_df.at["CBS", "CCSD(T)"] = corl_xtpl_helgaker_2(
        "ccsd(t)_dt",
        2,
        cbs_lot_df.at["aug-cc-pVDZ", "CCSD(T)"],
        3,
        cbs_lot_df.at["aug-cc-pVTZ", "CCSD(T)"],
    )

    # Sum up all energies to get NET CBS energy
    cbs_lot_df["NET"] = cbs_lot_df.sum(axis=1)

    return cbs_lot_df


def get_reference_be_result(
    bchmk_structs: dict[str, str], cbs_col: Dataset, cbs_list: List[str]
) -> pd.DataFrame:
    """
    Calculate reference energies for binding energy (be), interaction energy (ie),
    and deformation energy (de) for a set of benchmark structures.

    Parameters:
    bchmk_structs (dict[str, str]): Dictionary of benchmark structures.
    cbs_col (Dataset): Database collection for CBS energies.
    cbs_list (List[str]): List of CBS energy types.

    Returns:
    pd.DataFrame: DataFrame containing reference energies for 'be', 'ie', and 'de' for each benchmark structure.
    """
    logger = logging.getLogger("beep")
    df_dict = {
        "IE": None,
        "DE": None,
        "BE": None,
    }
    result_df = pd.DataFrame(columns=list(df_dict.keys()))

    for bench_struct in bchmk_structs:
        padded_log(logger, f"Calculating CBS extrapolations for {bench_struct}")
        logger.info(
            "\nInteraction Energy : IE\nDeformation Energy : DE\nBinding Energy : BE\n"
        )
        mol_name, surf_name, _ = bench_struct.split("_")
        struct_cbs_en = get_cbs_energy(cbs_col, bench_struct, cbs_list)
        mol_cbs_en = get_cbs_energy(cbs_col, mol_name.upper(), cbs_list)
        surf_cbs_en = get_cbs_energy(cbs_col, surf_name.upper(), cbs_list)
        struct_cbs_en_f1 = get_cbs_energy(cbs_col, bench_struct + "_f1", cbs_list)
        struct_cbs_en_f2 = get_cbs_energy(cbs_col, bench_struct + "_f2", cbs_list)

        ie = (
            struct_cbs_en - (struct_cbs_en_f1 + struct_cbs_en_f2)
        ) * qcel.constants.hartree2kcalmol
        be = (
            struct_cbs_en - (mol_cbs_en + surf_cbs_en)
        ) * qcel.constants.hartree2kcalmol
        de = (
            (struct_cbs_en_f1 + struct_cbs_en_f2) - (mol_cbs_en + surf_cbs_en)
        ) * qcel.constants.hartree2kcalmol

        logger.info(f"\nCCSD(T)/CBS result for structure: {bench_struct}")
        df_dict.update(
            {
                "IE": ie,
                "DE": de,
                "BE": be,
            }
        )
        for key, df in df_dict.items():
            logger.info(
                f"\nThe CCSD(T)/CBS incremental table for the {wstar} {key}{wstar} :"
            )
            # Replace NaN values with a dash
            df_formatted = df.fillna("-")
            logger.info(f"\n{df_formatted.to_string()}\n")

        # Extract the 'NET CBS' values and store them in a temporary dictionary
        temp_row = {key: df.loc["CBS", "NET"] for key, df in df_dict.items()}

        # Append the new row to the DataFrame with the identifier as the index
        result_df = pd.concat(
            [result_df, pd.DataFrame(temp_row, index=[bench_struct])],
            axis=0,
            join="outer",
        )
    padded_log(logger, "\n FINAL CCSD(T)/CBS RESULTS\n")
    logger.info(result_df)

    return result_df


def create_or_load_reaction_dataset(
    client, smol_name, surf_dset_name, bchmk_structs, dft_opt_lot, odset_dict
):
    """
    Create or update a ReactionDataset with benchmark structures and levels of theory.

    Parameters:
    - client (FractalClient): The active client connected to a QCFractal server.
    - smol_name (str): The name of the small molecule.
    - surf_dset_name (str): The name of the surface dataset.
    - bchmk_structs (list): A list of benchmark structures.
    - final_opt_lot (list): A list of levels of theory.
    - odset_dict (dict): A dictionary containing dataset information.

    Returns:
    - ds_be (ReactionDataset): The created or updated ReactionDataset object.
    """
    logger = logging.getLogger("beep")
    # Create or get benchmark binding energy dataset
    rdset_name = f"bchmk_be_{smol_name}_{surf_dset_name}"
    logger.info(f"Creating a loading ReactionDataset: {rdset_name}\n")
    try:
        ds_be = client.delete_collection("ReactionDataset", rdset_name)
    except KeyError:
        pass
    # try:
    #    ds_be = ReactionDataset(rdset_name, ds_type="rxn", client=client, default_program="psi4")
    #    ds_be.save()
    # except KeyError:
    #    ds_be = client.get_collection("ReactionDataset", rdset_name)

    # ds_be = client.get_collection("ReactionDataset", rdset_name)
    # Add reactions to the dataset
    ds_be = ReactionDataset(
        rdset_name, ds_type="rxn", client=client, default_program="psi4"
    )
    ds_be.save()
    n_entries = 0
    for bench_struct in bchmk_structs:
        for lot in dft_opt_lot:

            logger.info(f"Adding entry for {bench_struct} of {lot} geometry")
            be_stoich = create_be_stoichiometry(odset_dict, bench_struct, lot)
            bench_entry = f"{bench_struct}_{lot}"
            n_entries += 1
            try:
                ds_be.add_rxn(bench_entry, be_stoich)
            except KeyError:
                continue

    # Save changes to the dataset
    ds_be.save()
    logger.info(f"Created a total of {n_entries} in {rdset_name} {bcheck}")

    return ds_be


def create_be_stoichiometry(
    odset: OptimizationDataset, bench_struct: str, lot_geom: str
) -> dict:
    """
    Generates the Binding Energy (BE) stoichiometry for a given molecular system.

    This function takes a dataset, a benchmark structure identifier, and a level of theory for geometry optimization. It computes the BE stoichiometry for different scenarios, including the default BE stoichiometry, BE without counterpoise (nocp), interaction energy (ie), and deformation energy (de).

    Parameters:
    odset (Dataset): The dataset containing molecular records.
    bench_struct (str): A string identifier for the benchmark structure.
                        It should follow the format 'mol_name_surf_name_additionalInfo'.
    lot_geom (str): The level of theory used for geometry optimization in the dataset.

    Returns:
    dict: A dictionary containing different sets of tuples for BE stoichiometry calculations.
          Each tuple consists of a Molecule object and a corresponding coefficient.
          The keys of the dictionary represent different calculation scenarios:
          'default', 'be_nocp', 'ie', and 'de'.

    Notes:
    - The function assumes the input dataset (odset) is properly structured with necessary records.
    - The benchmark structure identifier (bench_struct) is expected to be in a specific format
      where the molecule name and surface name are separated by underscores.
    - The function uses the QCFractal interface for accessing and manipulating molecular data.

    """
    mol_name, surf_name, _ = bench_struct.split("_")
    bench_mol = (
        # odset[mol_name]
        # .get_record(name=mol_name, specification=lot_geom)
        odset[mol_name.upper()]
        .get_record(name=mol_name.upper(), specification=lot_geom)
        .get_final_molecule()
    )  ##  m_2

    bench_struc_mol = (
        odset[bench_struct]
        .get_record(name=bench_struct, specification=lot_geom)
        .get_final_molecule()
    )  ## mol, g, s
    bench_geom = bench_struc_mol.geometry.flatten()
    bench_symbols = bench_struc_mol.symbols

    surf_mod_mol = (
        odset[surf_name.upper()]
        .get_record(name=surf_name.upper(), specification=lot_geom)
        .get_final_molecule()
    )  ## m_1
    surf_symbols = surf_mod_mol.symbols

    f_bench_struc_mol = ptl.Molecule(
        symbols=bench_symbols,
        geometry=bench_geom,
        fragments=[
            list(range(0, len(surf_symbols))),
            list(range(len(surf_symbols), len(bench_symbols))),
        ],
    )

    j5 = f_bench_struc_mol.get_fragment(0)
    j4 = f_bench_struc_mol.get_fragment(1)
    j7 = f_bench_struc_mol.get_fragment(0, 1)
    j6 = f_bench_struc_mol.get_fragment(1, 0)

    be_stoic = {
        "default": [
            (f_bench_struc_mol, 1.0),
            (j4, 1.0),
            (j5, 1.0),
            (j7, -1.0),
            (j6, -1.0),
            (surf_mod_mol, -1.0),
            (bench_mol, -1.0),
        ],
        "be_nocp": [
            (f_bench_struc_mol, 1.0),
            (surf_mod_mol, -1.0),
            (bench_mol, -1.0),
        ],
        "ie": [(f_bench_struc_mol, 1.0), (j7, -1.0), (j6, -1.0)],
        "de": [(surf_mod_mol, -1.0), (bench_mol, -1.0), (j4, 1.0), (j5, 1.0)],
    }
    return be_stoic


def compute_be_dft_energies(
    ds_be, all_dft, basis="def2-tzvpd", program="psi4", tag="bench_dft"
):
    """
    Submits DFT computation jobs for Binding Energy (BE) calculations for various stoichiometries and functionals.

    Parameters:
    ds_be (DataSet): The QCFractal DataSet object for BE computations.
    all_dft (list): List of hybrid GGA functional names.
    basis (str, optional): Basis set to be used for DFT computations. Defaults to 'def2-tzvpd'.
    program (str, optional): Quantum chemistry program to use. Defaults to 'psi4'.
    tag (str, optional): Tag for categorizing the computation jobs. Defaults to 'ench_dft'.

    Returns:
    list: A list of computation IDs representing the submitted jobs.

    """

    logger = logging.getLogger("beep")
    stoich_list = ["default", "de", "ie", "be_nocp"]
    logger.info(
        f"Computing energies for the following stoichiometries: {' '.join(stoich_list)} (defualt = be)"
    )
    # logger.info(f"Sending DFT energy computations for the following functionals:")
    log_formatted_list(
        logger,
        all_dft,
        "Sending DFT energy computations for the following functionals:",
    )

    c_list_sub = []
    c_list_exis = []
    for i, func in enumerate(all_dft):
        c_per_func_sub = []
        c_per_func_exis = []
        for stoich in stoich_list:
            c = ds_be.compute(
                method=func,
                basis=basis,
                program=program,
                stoich=stoich,
                tag=tag,
            )
            c_list_sub.extend(list(c)[1][1])
            c_per_func_sub.extend(list(c)[1][1])
            c_list_exis.extend(list(c)[0][1])
            c_per_func_exis.extend(list(c)[0][1])
        logger.info(
            f"\n{func}: Existing {len(c_per_func_exis)}  Submitted {len(c_per_func_sub)}"
        )
        log_progress(logger, i, len(all_dft))

    logger.info(
        f"Submitted a total of {len(c_list_sub)} DFT computations. {len(c_list_exis)} are already computed"
    )
    return c_list_sub + c_list_exis


def check_jobs_status(
    client: FractalClient, job_ids: List[str], wait_interval: int = 600
) -> None:
    """
    Continuously monitors and reports the status of computations for given job IDs.

    Parameters:
    client (FractalClient): The client object used to interact with the QCFractal server.
    job_ids (List[str]): A list of job IDs whose status is to be checked.
    wait_interval (int): Interval in seconds between status checks.

    Returns:
    None: This function prints the status updates but does not return anything.
    """
    logger = logging.getLogger("beep")
    all_complete = False

    while not all_complete:
        status_counts = {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}

        job_stats = client.query_procedures(job_ids)
        for job in job_stats:
            if job:
                status = job.status.upper()
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    logger.info(f"Job ID {job_id}: Unknown status - {status}")
            else:
                logger.info(f"Job ID {job_id}: Not found in the database")

        # Log the status summary
        logger.info(
            f"Job Status Summary: {status_counts['INCOMPLETE']} INCOMPLETE, "
            f"{status_counts['COMPLETE']} COMPLETE, {status_counts['ERROR']} ERROR\n"
        )

        if status_counts["ERROR"] > 0:
            logger.info("Some jobs have ERROR status. Proceed with caution.")

        if status_counts["INCOMPLETE"] == 0:
            all_complete = True
            logger.info("All jobs are COMPLETE. Continuing with the execution.")
        else:
            time.sleep(wait_interval)  # Wait before the next check


def get_errors_dataframe(
    df: pd.DataFrame, ref_en_dict: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the absolute and relative errors between DataFrame values and reference energies.

    This function filters the DataFrame to include only rows where the index exists in the reference energy dictionary.
    It then computes the absolute and relative errors for each entry in the DataFrame compared to the corresponding
    reference energy.

    :param df: DataFrame containing the data to compare.
    :param ref_en_dict: Dictionary with reference energies. Keys should match the DataFrame index.
    :return: A tuple of two DataFrames (absolute error DataFrame, relative error DataFrame).
    """

    # Function to construct the key from the row index
    def construct_key(index: str) -> str:
        return "_".join(index.split("_")[:3])

    # Filter the DataFrame to only include rows where the index is present in ref_en_dict
    df = df[df.index.map(construct_key).isin(ref_en_dict.keys())]

    # Create new DataFrames to hold the absolute and relative errors
    abs_error_df = pd.DataFrame(index=df.index, columns=df.columns)
    rel_error_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Iterate over each row index and column
    for row_index in df.index:
        ref_value = ref_en_dict["_".join(row_index.split("_")[:3])]
        for col in df.columns:
            # Calculate the absolute error
            abs_error = df.at[row_index, col] - ref_value
            abs_error_df.at[row_index, col] = abs_error
            # Calculate the relative error
            rel_error_df.at[row_index, col] = abs_error / ref_value

    # The abs_error_df now contains the absolute errors, and rel_error_df contains the relative errors
    return abs_error_df, rel_error_df


def average_over_row(df, methods):
    # Drop columns with any NaN values
    df_cleaned = df.dropna(axis=1)

    # Initialize an empty dictionary to hold our averages
    averages = {}

    # Loop through each method and calculate the average for each column
    for method in methods:
        # Filter rows where the index contains the method
        method_df = df_cleaned.filter(like=method, axis=0)

        # Calculate mean for these rows and store in the dictionary
        averages[method] = method_df.mean()

    # Create a new DataFrame from the averages dictionary
    average_df = pd.DataFrame(averages).T  # Transpose to have methods as rows

    return average_df


def main():
    # Call the arguments
    args = parse_arguments()

    logger = setup_logging("bchmk_energy", args.molecule)
    logger.info(welcome_msg)

    client = ptl.FractalClient(
        address=args.client_address,
        verify=False,
        username=args.username,
        password=args.password,
    )

    # The name of the molecule to be sampled at level of theory opt_lot
    smol_name = args.molecule
    gr_method, gr_basis, gr_program = args.reference_geometry_level_of_theory
    geom_ref_opt_lot = gr_method + "_" + gr_basis

    bchmk_structs = args.benchmark_structures
    surf_dset_name = args.surface_model_collection
    smol_dset_name = args.small_molecule_collection
    dft_opt_lot = args.optimization_level_of_theory

    padded_log(logger, "Starting BEEP Energy benchmark procedure", padding_char=wstar)
    logger.info(f"Molecule: {smol_name}")
    logger.info(f"Surface Model: {smol_dset_name}")
    logger.info(f"Benchmark Structures: { ' '.join(bchmk_structs) }")
    logger.info(f"DFT and SQM  geometry levels of theory: {' '.join(dft_opt_lot)}")

    # Get multiplcity
    smol_dset = client.get_collection("OptimizationDataset", smol_dset_name)
    mol_mult = get_molecular_multiplicity(client, smol_dset, smol_name)
    logger.info(f"\nThe molecular multiplicity of {smol_name} is {mol_mult}\n\n")

    logger.info(
        f"Retriving data of the reference equilibirum geometries at {gr_method}/{gr_basis}:\n"
    )

    # Defining lists and Dictionaries
    odset_dict = {}
    bchmk_dset_names = {}

    # Save name of collections with the Benchmark strcutrues in a dictionary
    bchmk_dset_names = create_benchmark_dataset_dict(bchmk_structs)

    # Check existance of the collections
    check_collection_existence(client, *bchmk_dset_names.values())
    check_collection_existence(client, smol_dset_name)
    check_collection_existence(client, surf_dset_name)

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

    ## Retrive optimized molecules of the reference structures
    ref_geom_fmols = {}
    for struct_name, odset in odset_dict.items():
        print(struct_name)
        #record = odset.get_record(struct_name, specification="ccsd(t)_cc-pvtz")
        if struct_name in ["CO",  "W2", "W3"] :
            record = odset.get_record(struct_name, specification="ccsd(t)_cc-pvtz")
            ref_geom_fmols[struct_name] = record.get_final_molecule()
        else:
            record = odset.get_record(struct_name, specification=geom_ref_opt_lot)
            ref_geom_fmols[struct_name] = record.get_final_molecule()

    padded_log(logger, "CCSD(T)/CBS computations:")

    # Define CCSD(T)/CBS levels of theory
    cbs_list = [
        "scf_aug-cc-pVDZ",
        "scf_aug-cc-pVTZ",
        "scf_aug-cc-pVQZ",
        "mp2_aug-cc-pVQZ",
        "ccsd(t)_aug-cc-pVDZ",
        "ccsd(t)_aug-cc-pVTZ",
    ]

    log_formatted_list(
        logger,
        cbs_list,
        "Energies to compute for CCSD(T)/CBS (Semi-enlightend listing) : ",
    )

    # Creat Dataset collection for CCSD(T)/CBS calculation
    logger.info("\nCreating Dataset collection for CCSD(T)/CBS:")
    cbs_col = get_or_create_collection(
        client, "cbs" + "_" + smol_name + "_" + surf_dset_name, Dataset
    )

    # Populate dataset with structures and add keywords
    logger.info(
        f"\nAdding molecules and fragments to {cbs_col.name} Database collection:\n"
    )
    populate_dataset_with_structures(
        cbs_col, ref_geom_fmols, bchmk_structs, odset_dict, geom_ref_opt_lot
    )
    add_cc_keywords(cbs_col, mol_mult)

    ## Send al computations for CCSD(T)/CBS
    compute_all_cbs(cbs_col, cbs_list, mol_mult)

    ## Wait for CBS calculation completion
    check_dataset_status(cbs_col, cbs_list)

    # Assemble reference energy DF:
    ref_df = get_reference_be_result(bchmk_structs, cbs_col, cbs_list)
    logger.info(
        f"\nFinsihed the Calculation of the CCSD(T)/CBS reference energies:  {bcheck}\n"
    )

    # DFT energy computations

    padded_log(logger, "Initializing DFT Binding Energy Computations")

    # Load optimization levels of theory for BE computations
    dft_opt_lot = args.optimization_level_of_theory
    logger.info(
        f"The BE will be computed on the following Geometries: {' '.join(dft_opt_lot)}\n"
    )
    ds_be = create_or_load_reaction_dataset(
        client, smol_name, surf_dset_name, bchmk_structs, dft_opt_lot, odset_dict
    )

    # Compute BE for all functionals

    # dft_func = { "Meta_hybrid_gga" : meta_hybrid_gga() }
    dft_func = {
        "Hybrid GGA": hybrid_gga(),
        "Long range corrected": lrc(),
        "Meta Hybrid GGA": meta_hybrid_gga(),
    }

    for name, dft_f_list in dft_func.items():
        padded_log(
            logger,
            f"Sending computations for {name} functionals with a def2-tzvpd basis",
        )
        dft_ids = compute_be_dft_energies(
            ds_be, dft_f_list, basis="def2-tzvpd", program="psi4", tag="bench_en_dft"
        )
        check_jobs_status(client, dft_ids)

    # Create dataframe with results:
    ds_be._disable_query_limit = True
    ds_be.save()

    warnings.filterwarnings("ignore")

    padded_log(logger, "Retriving energies from ReactionDataset")
    df_be = ds_be.get_values(stoich="default").dropna(axis=1)
    df_ie = ds_be.get_values(stoich="ie").dropna(axis=1)
    df_de = ds_be.get_values(stoich="de").dropna(axis=1)

    df_be_ae, df_be_re = get_errors_dataframe(df_be, ref_df["BE"].to_dict())
    df_ie_ae, df_ie_re = get_errors_dataframe(df_ie, ref_df["IE"].to_dict())
    df_de_ae, df_de_re = get_errors_dataframe(df_de, ref_df["DE"].to_dict())

    # Define the folder path for 'json_data' in the current working directory
    folder_path_json = Path.cwd() / Path("en_json_data_" + smol_name)

    # Check if the folder exists, if not, create it
    if not folder_path_json.is_dir():
        folder_path_json.mkdir(parents=True, exist_ok=True)

    padded_log(logger, "Saving BE data in json files")
    # Save the DataFrames to JSON files in the 'json_data' folder
    df_be.to_json(folder_path_json / "BE_DFT.json", orient="index")
    logger.info(f"Saved BE dataframe in BE_DFT.json. {bcheck}\n")
    df_be_ae.to_json(folder_path_json / "BE_AE_DFT.json", orient="index")
    logger.info(f"Saved BE absolute error  dataframe in BE_AE_DFT.json. {bcheck}\n")
    df_be_re.to_json(folder_path_json / "BE_RE_DFT.json", orient="index")
    logger.info(f"Saved BE relative error  dataframe in BE_AE_DFT.json. {bcheck}\n")

    df_ie.to_json(folder_path_json / "IE_DFT.json", orient="index")
    logger.info(f"Saved IE dataframe in iE_DFT.json. {bcheck}\n")
    df_ie_ae.to_json(folder_path_json / "IE_AE_DFT.json", orient="index")
    logger.info(f"Saved IE absolute error  dataframe in IE_AE_DFT.json. {bcheck}\n")
    df_ie_re.to_json(folder_path_json / "IE_RE_DFT.json", orient="index")
    logger.info(f"Saved IE relative error  dataframe in IE_AE_DFT.json. {bcheck}\n")

    df_de.to_json(folder_path_json / "DE_DFT.json", orient="index")
    logger.info(f"Saved DE dataframe in DE_DFT.json. {bcheck}\n")
    df_de_ae.to_json(folder_path_json / "DE_AE_DFT.json", orient="index")
    logger.info(f"Saved DE absolute error  dataframe in DE_AE_DFT.json. {bcheck}\n")
    df_de_re.to_json(folder_path_json / "DE_RE_DFT.json", orient="index")
    logger.info(f"Saved DE relative error  dataframe in DE_AE_DFT.json. {bcheck}\n")

    # Create energy benchmark plots
    padded_log(logger, "Generating BE benchmark plots")

    # Define the folder path for 'json_data' in the current working directory
    folder_path_plots = Path.cwd() / Path("en_bchmk_plots_" + smol_name)

    # Check if the folder exists, if not, create it
    if not folder_path_plots.is_dir():
        folder_path_plots.mkdir(parents=True, exist_ok=True)

    # Reading plotting data from json

    df_be_plt = pd.read_json(folder_path_json / "BE_DFT.json", orient="index")
    df_be_ae_plt = pd.read_json(folder_path_json / "BE_AE_DFT.json", orient="index")

    df_de_re_plt = pd.read_json(folder_path_json / "DE_RE_DFT.json", orient="index")
    df_ie_re_plt = pd.read_json(folder_path_json / "IE_RE_DFT.json", orient="index")

    # Generating plots

    plot_violins(df_be_plt, bchmk_structs, smol_name, folder_path_plots, ref_df)
    padded_log(logger, "Generating violin plots")
    plot_density_panels(
        df_be_ae_plt, bchmk_structs, dft_opt_lot, smol_name, folder_path_plots
    )
    padded_log(logger, "Generating density plots")
    plot_mean_errors(
        df_be_ae_plt, bchmk_structs, dft_opt_lot, smol_name, folder_path_plots
    )
    padded_log(logger, "Generating MAE plots")
    plot_ie_vs_de(
        df_de_re_plt,
        df_ie_re_plt,
        bchmk_structs,
        dft_opt_lot,
        smol_name,
        folder_path_plots,
    )
    padded_log(logger, "Generating IE vs DE plots")

    # Log benchmark results
    padded_log(logger, "BINDING ENERGY BENCHMARK RESULTS", padding_char=gear)
    padded_log(logger, "BINDING ENERGY MAE")
    log_energy_mae(logger, df_be_ae)
    padded_log(logger, "INTERACTION ENERGY MAE")
    log_energy_mae(logger, df_ie_ae)
    padded_log(logger, "DEFORMATION ENERGY MAE")
    log_energy_mae(logger, df_de_ae)


if __name__ == "__main__":
    main()
