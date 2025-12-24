import qcfractal.interface as ptl
import logging
from typing import List, Tuple, Dict
from collections import Counter
from pydantic import ValidationError
from qcfractal.interface.client import FractalClient
from qcelemental.models.molecule import Molecule
from qcfractal.interface.collections import Dataset, OptimizationDataset, ReactionDataset

from .utils.logging_utils import *



def rmsd_filter(ds_opt, opt_lot: str, logger: logging.Logger) -> Dict[str, Molecule]:
    """
    Filters molecules based on their RMSD (Root Mean Square Deviation) values.

    Parameters:
    ds_opt (Dataset): Dataset object containing molecule records.
    opt_lot (str): Level of theory used for optimization.
    logger (logging.Logger): Logger instance for logging messages.

    Returns:
    Dict[str, Molecule]: Dictionary containing filtered molecule records.
    """
    logger.info("Starting rmsd_filter")
    molecules_to_delete: List[str] = []
    molecule_records: Dict[str, Molecule] = {}

    for index in ds_opt.df.index:
        try:
            # Retrieve the final molecule based on the specified level of theory
            molecule_records[index] = ds_opt.get_record(
                name=index, specification=opt_lot
            ).get_final_molecule()
        except (ValidationError, TypeError) as e:
            logger.warning(f"Error retrieving record {index}, Optimization finished with ERROR")
            continue

    molecule_keys: List[str] = list(molecule_records.keys())
    count = 0
    for i in range(len(molecule_keys)):
        for j in range(i + 1, len(molecule_keys)):
            count += 1
            mol1 = molecule_records[molecule_keys[i]]
            mol2 = molecule_records[molecule_keys[j]]
            # Calculate RMSD between two molecules
            rmsd = mol1.align(mol2, atoms_map=True)[1]["rmsd"]
            logger.debug(f"RMSD between {molecule_keys[i]} and {molecule_keys[j]}: {rmsd}")
            # If the RMSD is less than 0.25 and not zero, mark the second molecule for deletion
            if rmsd < 0.25 and rmsd != 0.0:
                if molecule_keys[j] not in molecules_to_delete:
                    molecules_to_delete.append(molecule_keys[j])

    logger.info(f"List of molecules to delete: {molecules_to_delete}")

    # Remove duplicates from the deletion list using a set
    unique_molecules_to_delete: List[str] = list(set(molecules_to_delete))

    # Remove marked molecules from the record dictionary
    for molecule_key in unique_molecules_to_delete:
        del molecule_records[molecule_key]

    logger.info(f"Remaining molecules after filtering: {list(molecule_records.keys())}")

    return molecule_records

def be_stoichiometry(smol_mol: Molecule, cluster_mol: Molecule, struc_mol: Molecule, logger: logging.Logger) -> Dict[str, List[Tuple[Molecule, float]]]:
    """
    Generates the Binding Energy (BE) stoichiometry for a given molecular system.

    This function computes the BE stoichiometry for different scenarios, including the 
    default BE stoichiometry, BE without counterpoise (nocp), interaction energy (ie), 
    and deformation energy (de).

    Parameters:
    smol_mol (Molecule): The small molecule bound to the surface.
    cluster_mol (Molecule): The surface or cluster model molecule.
    struc_mol (Molecule): The full structure with both the small molecule and cluster bound together.
    logger (logging.Logger): Logger instance for logging messages.

    Returns:
    Dict[str, List[Tuple[Molecule, float]]]: A dictionary containing different sets of tuples 
                                             for BE stoichiometry calculations.
                                             Each tuple consists of a Molecule object and 
                                             a corresponding coefficient.
                                             The keys of the dictionary represent different 
                                             calculation scenarios:
                                             'default', 'be_nocp', 'ie', and 'de'.

    Notes:
    - The function assumes the input molecules (smol_mol, cluster_mol, struc_mol) are valid.
    - The function uses the QCFractal interface for accessing and manipulating molecular data.
    """
    # Flatten the structure geometry and get the symbols
    geom = struc_mol.geometry.flatten()
    symbols = struc_mol.symbols
    surf_symbols = cluster_mol.symbols

    # Create a fragmented molecule with the surface as one fragment and the small molecule as another
    f_struc_mol = ptl.Molecule(
        symbols=symbols,
        geometry=geom,
        molecular_multiplicity=smol_mol.molecular_multiplicity,
        fragments=[
            list(range(0, len(surf_symbols))),
            list(range(len(surf_symbols), len(symbols))),
        ],
    )

    # Fragment extraction
    j5 = f_struc_mol.get_fragment(0)  # Surface fragment
    j4 = f_struc_mol.get_fragment(1)  # Small molecule fragment
    j7 = f_struc_mol.get_fragment(0, 1)  # Combined surface and small molecule
    j6 = f_struc_mol.get_fragment(1, 0)  # Alternative combined fragment

    logger.debug(f"Fragments generated: j4={j4}, j5={j5}, j6={j6}, j7={j7}")
    logger.debug(
    "Fragment multiplicities: "
    f"j4 (small molecule) = {j4.molecular_multiplicity}, "
    f"j5 (surface) = {j5.molecular_multiplicity}, "
    f"j6 (combined 1,0) = {j6.molecular_multiplicity}, "
    f"j7 (combined 0,1) = {j7.molecular_multiplicity}"
)

    # Binding energy stoichiometry dictionary
    be_stoic = {
        "default": [
            (f_struc_mol, 1.0),
            (j4, 1.0),
            (j5, 1.0),
            (j7, -1.0),
            (j6, -1.0),
            (cluster_mol, -1.0),
            (smol_mol, -1.0),
        ],
        "be_nocp": [
            (f_struc_mol, 1.0),
            (cluster_mol, -1.0),
            (smol_mol, -1.0),
        ],
        "ie": [(f_struc_mol, 1.0), (j7, -1.0), (j6, -1.0)],
        "de": [(cluster_mol, -1.0), (smol_mol, -1.0), (j4, 1.0), (j5, 1.0)],
    }

    return be_stoic

def create_or_load_reaction_dataset(
    client: FractalClient, 
    rdset_name: str, 
    opt_lot: str, 
    smol_mol: Molecule, 
    cluster_mol: Molecule, 
    ds_opt: Dataset, 
    opt_stru: Dict[str, object], 
    logger: logging.Logger
) -> ReactionDataset:
    """
    Create or update a ReactionDataset with benchmark structures and levels of theory.

    Parameters:
    - client (FractalClient): The active client connected to a QCFractal server.
    - rdset_name (str): The name of the ReactionDataset to create or update.
    - opt_lot (str): The level of theory used for optimizations.
    - smol_mol (Molecule): The small molecule to be included in the dataset.
    - cluster_mol (Molecule): The cluster (surface) molecule to be included.
    - ds_opt (Dataset): Dataset object containing the optimization records.
    - opt_stru (dict): A dictionary containing the structure identifiers as keys.
    - logger (logging.Logger): Logger instance for logging messages.

    Returns:
    - ReactionDataset: The created or updated ReactionDataset object.
    """
    # Try to delete the dataset if it already exists
    try:
        client.delete_collection("ReactionDataset", rdset_name)
    except KeyError:
        pass  # If dataset doesn't exist, continue without error

    # Create a new ReactionDataset
    ds_be = ReactionDataset(rdset_name, ds_type="rxn", client=client, default_program="psi4")
    ds_be.save()

    # Retrieve the ReactionDataset after saving
    ds_be = client.get_collection("ReactionDataset", rdset_name)

    n_entries = 0
    padded_log(logger, f"Populating the dataframe with {len(opt_stru.keys())} new entries", padding_char="*", total_length=60)

    # Iterate over the structures in the optimization dataset
    for st in opt_stru.keys():
        logger.info(f"Processing structure: {st}")
        rr = ds_opt.get_record(st, opt_lot)

        # Skip structures with optimization errors
        if rr.status == "ERROR":
            logger.warning(f"WARNING: Optimization of {st} with {opt_lot} finished with error. Will skip this structure.")
            continue

        # Get the final optimized molecule
        struct_mol = rr.get_final_molecule()

        # Generate binding energy stoichiometry
        logger.info(f"Generating BE stoichiometry for {st}")
        be_stoich = be_stoichiometry(smol_mol, cluster_mol, struct_mol, logger)

        # Add the reaction to the ReactionDataset
        rds_entry = f"{st}"
        n_entries += 1
        try:
            ds_be.add_rxn(rds_entry, be_stoich)
            logger.info(f"Successfully added {st} to the dataset.\n")
        except KeyError:
            logger.warning(f"Failed to add {st}. Skipping entry.")
            continue

    # Save changes to the dataset
    ds_be.save()

    logger.info(f"Created a total of {n_entries} entries in {rdset_name}.\n")

    return ds_be


def compute_be_dft_energies(
    ds_be,
    all_dft: List[str],
    tag: str,
    program: str,
    logger: logging.Logger,
    keyword: str = None
) -> List[str]:
    """
    Submits DFT computation jobs for Binding Energy (BE) calculations for various stoichiometries and functionals.

    Parameters:
    - ds_be (Dataset): The QCFractal Dataset object for BE computations.
    - all_dft (list): List of hybrid GGA functional names.
    - tag (str): Tag for the QCFractal manager computation.
    - program (str): Quantum chemistry program to use for the computations.
    - logger (logging.Logger): Logger instance for logging messages.
    - keyword (str, optional): Keyword to use in the computation. Defaults to None.

    Returns:
    - list: A list of computation IDs representing the submitted jobs.
    """
    stoich_list = ["default", "de", "ie", "be_nocp"]
    logger.info(f"Computing energies for the following stoichiometries: {' '.join(stoich_list)} (default = be)\n")

    log_formatted_list(logger, all_dft, "Sending DFT energy computations for the following Levels of theory:", max_rows=1)

    c_list_sub = []
    c_list_exis = []

    logger.info(f"\nSending DFT computations with tag: {tag}\n")

    for i, lot in enumerate(all_dft):
        method, basis = lot.split("_")  # Split method and basis
        logger.info(f"Processing method: {method}, basis: {basis}")

        c_per_lot_sub = []
        c_per_lot_exis = []

        for stoich in stoich_list:
            c = ds_be.compute(
                method=method,
                basis=basis,
                program=program,
                stoich=stoich,
                tag=tag,
                keywords=keyword,
            )
            # Extract submitted and existing computations
            c_list_sub.extend(list(c)[1][1])
            c_per_lot_sub.extend(list(c)[1][1])
            c_list_exis.extend(list(c)[0][1])
            c_per_lot_exis.extend(list(c)[0][1])

        logger.info(f"{lot}: Existing {len(c_per_lot_exis)}  Submitted {len(c_per_lot_sub)}")

    logger.info(f"\nSubmitted a total of {len(c_list_sub)} DFT computations. {len(c_list_exis)} are already computed.")

    return c_list_sub + c_list_exis


def compute_hessian(
    client: FractalClient,
    ds_be_name: str,
    opt_lot: str,
    mult: int,
    hess_tag: str,
    logger: logging.Logger,
    program: str = 'psi4'
) -> List[str]:
    """
    Compute hessian for all molecules in a given ReactionDataset collection.

    Parameters:
    - client (FractalClient): A connection to the QCFractal server.
    - ds_be_name (str): The name of the ReactionDataset collection from which to compute hessians.
    - opt_lot (str): The level of theory, split into method and basis (e.g., 'b3lyp_def2svp').
    - mult (int): The multiplicity of the molecules.
    - hess_tag (str): The tag used for the compute submission.
    - logger (logging.Logger): Logger instance for logging messages.
    - program (str, optional): The quantum chemistry program to use. Defaults to 'psi4'.

    Returns:
    - list: A combined list of submitted and existing computations.
    """

    # Fetch the ReactionDataset collection by name
    try:
        ds_be = client.get_collection("ReactionDataset", ds_be_name)
    except KeyError:
        logger.info(f"\nWARNING: Reaction database {ds_be_name} does not exist.\n")
        return []

    padded_log(logger, f"Computing Hessians for {ds_be.name}", padding_char="*", total_length=60)

    # Extract method and basis from opt_lot
    method, basis = opt_lot.split("_")

    df_all = ds_be.get_entries()
    mols = df_all[df_all['stoichiometry'] == 'be_nocp']['molecule']
    u_mols = list(set(mols))

    # Special case for atoms: removes id if its an atom
    mol_list = client.query_molecules(u_mols)
    for m in mol_list:
        if len(m.symbols) == 1:
            logger.info(f"\nAtom id {m.id} removed")
            u_mols.remove(m.id)

    log_formatted_list(logger, u_mols, "Sending Hessian computations for the following molecules:", max_rows=5)

    # Set keywords depending on the multiplicity
    logger.info(f"\nWill compute Hessian at {method}/{basis} level of theory")
    if mult == 2:
        kw = ptl.models.KeywordSet(
            **{"values": {"function_kwargs": {"dertype": 1}, "reference": "uks"}}
        )
        #method = method[1:]  # Adjust method for unrestricted cases
    else:
        kw = ptl.models.KeywordSet(
            **{"values": {"function_kwargs": {"dertype": 1}}}
        )

    logger.info(f"\nComputing Hessian at {method}/{basis} level of theory")
    logger.info(f"Using keywords: {kw.values}")

    kw_id = client.add_keywords([kw])[0]
    c = client.add_compute(
        program, method, basis, "hessian", kw_id, u_mols, tag=hess_tag
    )

    # List operations to track submissions and existing computations
    c_list_sub = list(c)[1][1]
    c_list_exis = list(c)[0][1]

    logger.info(
        f"\nExisting: {len(c_list_exis)} Submitted: {len(c_list_sub)}"
    )
    logger.info(
        f"\nSubmitted a total of {len(c_list_sub)} computations. "
        f"{len(c_list_exis)} are already computed."
    )

    return c_list_sub + c_list_exis

