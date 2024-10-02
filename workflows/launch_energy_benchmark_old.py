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
    #parser.add_argument(
    #    "--reference-energy-level-of-theory",
    #    nargs=3,
    #    default=["ccsd(t)", "cbs", "psi4"],
    #    help="The level of theory and program for the geometry refinement in the format: method_basis (default: df-ccsd(t) cc-pvdz molpro)",
    #)
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
        logger.info(f"Collection of type {collection_type.__name__} with name {dset_name} already exists. {bcheck}\n")
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
    geom_ref_opt_lot: str
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
                .get_record(name=surf_name.upper(), specification=geom_ref_opt_lot)
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
        logger.info(f"\nCreating keywords for closed shell coupled cluster computation")
        kw_dict = {"scf_type": "df", "cc_type": "df", "freeze_core": "true"}
        logger.info(f"Keywords dictionary: {kw_dict}")
        kw_dfit = ptl.models.KeywordSet(
            values = kw_dict
        )
    elif mol_mult == 2:
        logger.info(f"Creating keywords for open shell coupled cluster computation")
        kw_dict = {"reference": "uhf", "freeze_core": "true"}
        logger.info(f"Keywords dictionary: {kw_dict}")
        kw_dfit = ptl.models.KeywordSet(
            values=kw_dict
        )

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
    for lot in cbs_list:
        # Splitting the level of theory into method and basis
        method, basis = lot.split("_")[0], lot.split("_")[1]

        # Tag assignment based on molecule multiplicity
        tag = "cbs_en_radical" if mol_mult == 2 else "cbs_en"

        # Check if "scf" is not in the level of theory
        if "scf" not in lot:
            c = cbs_col.compute(
                method, basis, tag=tag, keywords="df", program="psi4"
            )
        else:
            c = cbs_col.compute(
                method, basis, tag=tag, program="psi4"
            )

        all_cbs_ids.extend(c.ids)
        logger.info(f"This is the compute return {c}")

    return all_cbs_ids



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
            if not "scf" in method:
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


def calculate_reference_energies(bchmk_structs: dict[str, str], cbs_col: Dataset, cbs_list: List[str]) -> dict[str, dict]:
    """
    Calculate reference energies for binding energy (be), ionization energy (ie),
    and dissociation energy (de) for a set of benchmark structures.

    Parameters:
    bchmk_structs (list): List of benchmark structures.
    cbs_col (collection): Database collection for CBS energies.
    cbs_list (list): List of CBS energy types.

    Returns:
    dict: Dictionary containing reference energies for 'be', 'ie', and 'de'.
    """
    logger = logging.getLogger("beep")
    ref_en_dict = {"be": None, "ie": None, "de": None}
    for en in ref_en_dict.keys():
        ref_en_struct = {}
        for bench_struct in bchmk_structs:
            mol_name, surf_name, _ = bench_struct.split("_")
            mol_cbs_en = get_cbs_energy(cbs_col, mol_name.upper(), cbs_list)
            surf_cbs_en = get_cbs_energy(cbs_col, surf_name.upper(), cbs_list)
            struct_cbs_en = get_cbs_energy(cbs_col, bench_struct, cbs_list)
            struct_cbs_en_f1 = get_cbs_energy(cbs_col, bench_struct + "_f1", cbs_list)
            struct_cbs_en_f2 = get_cbs_energy(cbs_col, bench_struct + "_f2", cbs_list)

            # Calculate interaction energy (ie)
            ie = (struct_cbs_en - (struct_cbs_en_f1 + struct_cbs_en_f2)) * qcel.constants.hartree2kcalmol

            # Calculate binding energy (be)
            be = (struct_cbs_en - (mol_cbs_en + surf_cbs_en)) * qcel.constants.hartree2kcalmol

            # Calculate deformation energy (de)
            de = (((struct_cbs_en_f1 + struct_cbs_en_f2) - (mol_cbs_en + surf_cbs_en)) * qcel.constants.hartree2kcalmol) * (-1)

            if en == "be":
                ref_en_struct[bench_struct] = be
            elif en == "ie":
                ref_en_struct[bench_struct] = ie
            elif en == "de":
                ref_en_struct[bench_struct] = de

        ref_en_dict[en] = ref_en_struct

    return ref_en_dict





def create_be_stoichiometry(odset, bench_struct, lot_geom):
    mol_name, surf_name, _ = bench_struct.split("_")
    bench_mol = (
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
        "de": [(surf_mod_mol, 1.0), (bench_mol, 1.0), (j4, -1.0), (j5, -1.0)],
    }
    return be_stoic


def get_cbs_energy(ds: Dataset, struct, cbs_lot_list):
    cbs_lot_en = {}
    for lot in cbs_lot_list:
        method, basis = lot.split("_")
        if not "ccsd" in method:
            rec = ds.get_records(method=method, basis=basis, program="Psi4").loc[
                struct
            ][0]
            cbs_lot_en[lot] = rec.return_result
        else:
            rec = ds.get_records(
                method=method, basis=basis, program="Psi4", keywords="df"
            ).loc[struct][0]
            cbs_lot_en["mp2_" + basis] = rec.dict()["extras"]["qcvars"][
                "MP2 TOTAL ENERGY"
            ]
            cbs_lot_en["ccsd_" + basis] = rec.dict()["extras"]["qcvars"][
                "CCSD TOTAL ENERGY"
            ]
            cbs_lot_en[lot] = rec.return_result

    ## Extrapolations using the dictionary:
    scf_dtq = scf_xtpl_helgaker_3(
        "scf_ext",
        2,
        cbs_lot_en["scf_aug-cc-pVDZ"],
        3,
        cbs_lot_en["scf_aug-cc-pVTZ"],
        4,
        cbs_lot_en["scf_aug-cc-pVQZ"],
    )
    scf_tq = scf_xtpl_helgaker_2(
        "scf_ext1", 3, cbs_lot_en["scf_aug-cc-pVTZ"], 4, cbs_lot_en["scf_aug-cc-pVQZ"]
    )
    mp2_dt = corl_xtpl_helgaker_2(
        "mp2_ext", 2, cbs_lot_en["mp2_aug-cc-pVDZ"], 3, cbs_lot_en["mp2_aug-cc-pVTZ"]
    )
    mp2_tq = corl_xtpl_helgaker_2(
        "mp2_ext", 3, cbs_lot_en["mp2_aug-cc-pVTZ"], 4, cbs_lot_en["mp2_aug-cc-pVQZ"]
    )
    ccsd_dt = corl_xtpl_helgaker_2(
        "xtpl", 2, cbs_lot_en["ccsd_aug-cc-pVDZ"], 3, cbs_lot_en["ccsd_aug-cc-pVTZ"]
    )
    ccsdt_dt = corl_xtpl_helgaker_2(
        "xtpl",
        2,
        cbs_lot_en["ccsd(t)_aug-cc-pVDZ"],
        3,
        cbs_lot_en["ccsd(t)_aug-cc-pVTZ"],
    )

    ## Contributions:

    mp2_cbs_corr = mp2_tq - scf_tq
    ccsd_cbs_corr = ccsd_dt - mp2_dt
    ccsdt_cbs_corr = ccsdt_dt - ccsd_dt

    return scf_dtq + mp2_cbs_corr + ccsd_cbs_corr + ccsdt_cbs_corr


def abs_error_dataframe(df, ref_en_dict):
    # Create a new DataFrame to hold the results
    result_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Iterate over each row index and column, subtract, and take the absolute value
    for row_index in df.index:
        for col in df.columns:
            dict_key = (
                row_index.split("_")[0]
                + "_"
                + row_index.split("_")[1]
                + "_"
                + row_index.split("_")[2]
            )  # Construct the key from the row index
            if dict_key in ref_en_dict:
                # Subtract the dictionary value from the DataFrame entry and take the absolute value
                result_df.at[row_index, col] = abs(
                    df.at[row_index, col] - ref_en_dict[dict_key]
                )

    # The result_df now contains the absolute differences
    return result_df


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


def save_df_to_json(df, filename):
    """
    Save a pandas DataFrame to a JSON file.

    Parameters:
    df (pandas.DataFrame): The DataFrame to save.
    filename (str): The name of the file where the DataFrame will be saved.
    """
    try:
        # df.to_json(filename, orient='records', lines=True)
        df.to_json(filename)
        print(f"DataFrame successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving DataFrame to JSON: {e}")


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
    gr_method, gr_basis, gr_program  = args.reference_geometry_level_of_theory
    geom_ref_opt_lot = gr_method + "_" + gr_basis

    bchmk_structs = args.benchmark_structures
    surf_dset_name = args.surface_model_collection
    smol_dset_name = args.small_molecule_collection
    dft_opt_lot = args.optimization_level_of_theory

    padded_log(logger, "Starting BEEP Energy benchmark procedure", padding_char=gear)
    logger.info(f"Molecule: {smol_name}")
    logger.info(f"Surface Model: {smol_dset_name}")
    logger.info(f"Benchmark Structures: {bchmk_structs}")
    logger.info(f"DFT and SQM  geometry levels of theory: {dft_opt_lot}")
    
    # Defining lists and Dictionaries
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
    logger.info(f"\n The molecular multiplicity of {smol_name} is {mol_mult}\n")

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

    ## Specification for reference of geometry benchmark:

    #padded_log(logger, "Start of the geometry refrence processing")
    #logger.info(f"Method: {gr_method}")
    #logger.info(f"Basis: {gr_basis}")
    #logger.info(f"Program: {gr_program}\n")
    #for odset in odset_dict.values():
    #    create_and_add_specification(
    #        odset,
    #        method=gr_method,
    #        basis=gr_basis,
    #        program=gr_program,
    #        geom_keywords=None,
    #        qc_keywords=None,
    #    )

    ## Optimize all three molecules at the reference benchmark level of theory
    #ct = 0
    #for struct_name, odset in odset_dict.items():
    #    ct += optimize_reference_molecule(
    #        odset, struct_name, geom_ref_opt_lot, mol_mult
    #    )

    #logger.info(f"\nSend a total of {ct} structures to compute at the {geom_ref_opt_lot} level of theory\n")

    ### Wait for the Opimizations to finish
    #wait_for_completion(odset_dict, geom_ref_opt_lot, wait_interval=600, check_errors=True)

    #padded_log(logger, "Start of the DFT geometry computations")

    ### Optimize with DFT functionals
    #dft_program = args.dft_optimization_program

    ### Saving funcitona lists in a dictionary
    #dft_geom_functionals = {
    #    "geom_hmgga_dz": geom_hmgga_dz(),
    #    "geom_hmgga_tz": geom_hmgga_tz(),
    #    "geom_gga_dz": geom_gga_dz(),
    #    "geom_sqm_mb": geom_sqm_mb(),
    #}

    ## Combine all functionals into one list
    #all_dft_functionals = [
    #    functional
    #    for functionals in dft_geom_functionals.values()
    #    for functional in functionals
    #]
   
    #logger.info(f"Program: {dft_program}")
    #logger.info(f"DFT and SQM geometry methods:")
    #dict_to_log(logger, dft_geom_functionals)
    ## Sending all DFT geometry optimizations
    #ct = 0
    #c = 0
    #padded_log(logger,'Start sending DFT optimizations')
    #for struct_name, odset in odset_dict.items():
    #    logger.info(f"\nSending geometry optimizations for {struct_name}")
    #    cs = 0
    #    for functionals in dft_geom_functionals.values():
    #        for functional in functionals:
    #            method, basis = functional.split("_")
    #            cs += optimize_dft_molecule(odset, struct_name, method, basis, dft_program)
    #            ct += cs
    #            c+=1
    #    logger.info(f"Send {cs} geometry optimizations for structure {struct_name}")

    #logger.info(f"\nSend {ct}/{c} to the tag bench_dft"  )

    #wait_for_completion(
    #    odset_dict, all_dft_functionals, wait_interval=200, check_errors=False
    #)

    ## Retrive optimized molecules of the reference structures
    ref_geom_fmols = {}
    for struct_name, odset in odset_dict.items():
        record = odset.get_record(struct_name, specification=geom_ref_opt_lot)
        ref_geom_fmols[struct_name] = record.get_final_molecule()

    #padded_log(logger, "Start of RMSD comparsion between DFT and {} geometries", geom_ref_opt_lot)
    ## Compare RMSD for all functional groups
    #best_opt_lot, rmsd_df = compare_all_rmsd(
    #    dft_geom_functionals, odset_dict, ref_geom_fmols
    #)
    ## Logging the results of the geometry benchmark
    #padded_log(logger, "BENCHMARK RESULSTS")
    #log_dataframe_averages(logger, rmsd_df)

    ## Plot Geomtry benchmark results
    #rmsd_histograms(rmsd_df)
    #padded_log(logger, "Geometry Benchmark finished successfully! Hasta pronto!", padding_char=mia0911)

    ## END GEOMETRY BENCHMARK, INCLUDE SOME RMSD PLOTS IN THE FUTURE.
    #return None

    # Creat Dataset collection for CCSD(T)/CBS calculation
    cbs_col = get_or_create_collection(
        client, "cbs" + "_" + smol_name + "_" + surf_dset_name, Dataset
    )


    # Populate dataset with structures and add keywords
    populate_dataset_with_structures(cbs_col, ref_geom_fmols, bchmk_structs, odset_dict, geom_ref_opt_lot)
    add_cc_keywords(cbs_col, mol_mult)

    # Define CCSD(T)/CBS levels of theory
    cbs_list = [
        "scf_aug-cc-pVDZ",
        "scf_aug-cc-pVTZ",
        "scf_aug-cc-pVQZ",
        "mp2_aug-cc-pVQZ",
        "ccsd(t)_aug-cc-pVDZ",
        "ccsd(t)_aug-cc-pVTZ",
    ]

    
    compute_all_cbs(cbs_col, cbs_list, mol_mult)
    #Compute all CBS 
    #all_cbs_ids = []
    #for lot in cbs_list:
    #    if not "scf" in lot:
    #        if mol_mult == 1:
    #            c = cbs_col.compute(
    #                lot.split("_")[0],
    #                lot.split("_")[1],
    #                tag="cbs_en",
    #                keywords="df",
    #                program="psi4",
    #            )
    #            all_cbs_ids.extend(c.ids)
    #        elif mol_mult == 2:
    #            c = cbs_col.compute(
    #                lot.split("_")[0],
    #                lot.split("_")[1],
    #                tag="cbs_en_radical",
    #                keywords="df",
    #                program="psi4",
    #            )
    #            all_cbs_ids.extend(c.ids)
    #    else:
    #        if mol_mult == 1:
    #            c = cbs_col.compute(
    #                lot.split("_")[0], lot.split("_")[1], tag="cbs_en", program="psi4"
    #            )
    #        elif mol_mult == 2:
    #            c = cbs_col.compute(
    #                lot.split("_")[0],
    #                lot.split("_")[1],
    #                tag="cbs_en_radical",
    #                program="psi4",
    #            )
    #            all_cbs_ids.extend(c.ids)

    ## Wait for CBS calculation completion
    check_dataset_status(client, cbs_col, cbs_list)
    return None

    # Get reference energy dict:
    ref_en_dict = {"be": None, "ie": None, "de": None}
    for en in ref_en_dict.keys():
        ref_en_struct = {}
        for bench_struct in bchmk_structs:
            mol_name, surf_name, _ = bench_struct.split("_")
            mol_cbs_en = get_cbs_energy(cbs_col, mol_name.upper(), cbs_list)
            surf_cbs_en = get_cbs_energy(cbs_col, surf_name.upper(), cbs_list)
            struct_cbs_en = get_cbs_energy(cbs_col, bench_struct, cbs_list)
            struct_cbs_en_f1 = get_cbs_energy(cbs_col, bench_struct + "_f1", cbs_list)
            struct_cbs_en_f2 = get_cbs_energy(cbs_col, bench_struct + "_f2", cbs_list)
            ie = (
                struct_cbs_en - (struct_cbs_en_f1 + struct_cbs_en_f2)
            ) * qcel.constants.hartree2kcalmol  # * (-1)
            be = (
                struct_cbs_en - (mol_cbs_en + surf_cbs_en)
            ) * qcel.constants.hartree2kcalmol  # * (-1)
            de = (
                ((struct_cbs_en_f1 + struct_cbs_en_f2) - (mol_cbs_en + surf_cbs_en))
                * qcel.constants.hartree2kcalmol
                * (-1)
            )
            if en == "be":
                ref_en_struct[bench_struct] = be
            elif en == "ie":
                ref_en_struct[bench_struct] = ie
            elif en == "de":
                ref_en_struct[bench_struct] = de
        ref_en_dict[en] = ref_en_struct
    print(ref_en_dict)

    # Create or get bench_be dataset
    rdset_name = "bchmk_be_" + smol_name + "_" + surf_dset_name
    try:
        ds_be = ReactionDataset(
            rdset_name, ds_type="rxn", client=client, default_program="psi4"
        )
        ds_be.save()
    except KeyError:
        ds_be = client.get_collection("ReactionDataset", rdset_name)

    for bench_struct in bchmk_structs:
        for lot in final_opt_lot:
            be_stoich = create_be_stoichiometry(odset_dict, bench_struct, lot)
            bench_entry = bench_struct + "_" + lot
            try:
                ds_be.add_rxn(bench_entry, be_stoich)
            except KeyError:
                continue
    ds_be.save()

    # Compute BE for all functionals

    hybrid_gga = hybrid_gga()
    lrc = lrc()
    meta_hybrid_gga = meta_hybrid_gga()

    all_dft = hybrid_gga + lrc + meta_hybrid_gga
    stoich_list = ["default", "de", "ie", "be_nocp"]

    # Send DFT Jobs
    c_list = []
    for func in all_dft:
        for stoich in stoich_list:
            c = ds_be.compute(
                method=func,
                basis="def2-tzvp",
                program="psi4",
                stoich=stoich,
                tag="bench_dft",
            )
            c_list.extend(c)
    print(f"Sumbited a total of {len(c_list)} DFT computations")

    # Check job completion for the ReactionDataset

    return None

    # Create dataframe with results:
    ds_be._disable_query_limit = True
    ds_be.save()

    df_be = ds_be.get_values(stoich="default")
    be_ae = abs_error_dataframe(df_be, ref_en_dict["be"])
    f_df_ae_be = average_over_row(be_ae, list(final_opt_lot.keys()))

    df_ie = ds_be.get_values(stoich="ie")
    ie_ae = abs_error_dataframe(df_ie, ref_en_dict["ie"])
    f_df_ae_ie = average_over_row(ie_ae, list(final_opt_lot.keys()))

    df_de = ds_be.get_values(stoich="de")
    de_ae = abs_error_dataframe(df_de, ref_en_dict["de"])
    f_df_ae_de = average_over_row(de_ae, list(final_opt_lot.keys()))
    print(f_df_ae_de)

    save_df_to_json(f_df_ae_be, "be_ae.json")
    save_df_to_json(f_df_ae_ie, "be_ie.json")
    save_df_to_json(f_df_ae_de, "be_de.json")

    save_df_to_json(df_be, "be_dft.json")
    save_df_to_json(df_ie, "ie_dft.json")
    save_df_to_json(df_de, "de_dft.json")

    for st in ref_en_dict["be"].keys():
        plot_violin_with_ref(st, ref_en_dict["be"], df_be.T.dropna(axis=0))


if __name__ == "__main__":
    main()
