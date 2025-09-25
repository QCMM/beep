import sys
import time
import logging
import numpy as np
from pathlib import Path
from qcfractal.interface.collections.optimization_dataset import OptimizationDataset
from qcelemental.models.molecule import Molecule
from qcfractal.interface.client import FractalClient
from typing import List, Tuple, Optional
from collections import Counter
from .molecule_sampler import random_molecule_sampler as mol_sample


def generate_shell_list(sampling_shell: float, condition: str) -> List[float]:
    """
    Generate a list of sampling shells based on the given condition.

    Parameters:
    - sampling_shell (float): The primary sampling shell value.
    - condition (str): The condition to adjust the sampling shell list.
                       It can be one of 'sparse', 'normal', or 'fine'.

    Returns:
    - List[float]: A list containing the adjusted sampling shell values.

    Raises:
    - ValueError: If the condition is not one of ['sparse', 'normal', 'fine'].

    Examples:
    >>> generate_shell_list(10.0, 'sparse')
    [10.0]

    >>> generate_shell_list(10.0, 'normal')
    [10.0, 8.0, 12.0]

    >>> generate_shell_list(10.0, 'fine')
    [10.0, 7.5, 9.0, 11.0, 15.0]
    """

    conditions_map = {
        "sparse": [sampling_shell],
        "normal": [sampling_shell, sampling_shell * 0.8, sampling_shell * 1.2],
        "fine": [
            sampling_shell,
            sampling_shell * 0.8,
            sampling_shell * 1.2,
            sampling_shell * 0.75,
            sampling_shell * 1.5,
        ],
        "hyperfine": [
            sampling_shell,
            sampling_shell * 0.8,
            sampling_shell * 1.2,
            sampling_shell * 0.75,
            sampling_shell * 1.5,
            sampling_shell * 0.9,
            sampling_shell * 1.1,
        ],
    }

    shell_list = conditions_map.get(condition)
    if shell_list is None:
        raise ValueError(
            "Condition should be one of ['sparse', 'normal', 'fine', 'hyperfine']"
        )

    return shell_list


def get_job_ids(
    ds_opt: OptimizationDataset, entry_list: List[str], opt_lot: str
) -> List[int]:
    pid_list = []
    for n in entry_list:
        opt_rec = ds_opt.get_record(n, opt_lot)
        pid_list.append(opt_rec.id)
    return pid_list


def check_for_completion(
    client: "FractalClient", pid: List[str], frequency: int = 60
) -> Tuple[bool, Counter]:
    """
    Checks the status of jobs and returns their completion status and counts of each status.

    Parameters:
    - client: An instance of FractalClient.
    - pid: A list of job IDs.
    - frequency: Time in seconds to wait before checking again. Default is 60 seconds.

    Returns:
    - Tuple of:
      - True if all jobs are complete, False otherwise.
      - Counter object with counts of each status.
    """
    status = []
    if pid:
        status = [r.status for r in client.query_procedures(pid)]
    status_counts = Counter(status)

    # Check if all jobs have the status "ERROR"
    # if len(status) == status_counts["ERROR"]:
    #    raise RuntimeError(
    #        "All jobs have the status 'ERROR'. Please delete the OptimizationDataset and check your Specification"
    #    )

    all_jobs_complete = "INCOMPLETE" not in status

    return all_jobs_complete, status_counts


def wait_for_completion(
    client: FractalClient, pid_list: List[str], FREQUENCY: int, logger: logging.Logger,
) -> None:
    jobs_complete = False
    logger.info("Checking for job completion....")
    while not jobs_complete:
        jobs_complete, counts = check_for_completion(client, pid_list, FREQUENCY)
        status_str = " ".join([f"{s}: {count}, " for s, count in counts.items()])
        logger.info("The status of the Optimization jobs: " + status_str)
        if not jobs_complete:
            time.sleep(FREQUENCY)


def get_opt_molecules(
    ds_opt: "OptimizationDataset",
    entry_list: List[str],
    opt_lot: str,
    status: Optional[str] = "COMPLETE",
) -> List[Tuple[str, Molecule]]:
    """
    Retrieves records from the dataset based on the provided status.

    Parameters:
    - ds_opt: The dataset object with the `get_record` method.
    - entry_list: A list of entries to query.
    - opt_lot: The optimization level of theory or similar parameter.
    - status: Desired status of the records to retrieve (default is "COMPLETE").

    Returns:
    - A list of records that match the specified status.
    """
    mol_list = []
    for n in entry_list:
        record = ds_opt.get_record(n, opt_lot)
        if record.status == status:
            mol_list.append((n, record.get_final_molecule()))
    return mol_list


def compute_rmsd(
    mol1: Molecule, mol2: Molecule, rmsd_symm: bool
) -> Tuple[float, float]:
    rmsd_val_mirror = 10.0
    if rmsd_symm:
        align_mols_mirror = mol1.align(mol2, atoms_map=True, run_mirror=True)
        rmsd_val_mirror = align_mols_mirror[1]["rmsd"]
    align_mols = mol1.align(mol2, atoms_map=True)
    rmsd_val = align_mols[1]["rmsd"]
    return rmsd_val, rmsd_val_mirror


def filter_binding_sites(
    mol_list1: List[Tuple[str, Molecule]],
    mol_list2: List[Tuple[str, Molecule]],
    cut_off_val: float,
    rmsd_symm: bool,
    logger: logging.Logger,
) -> List[Molecule]:
    """
    Filters out duplicate binding sites based on RMSD (Root Mean Square Deviation) values between molecules.

    This function compares molecules in `mol_list1` and `mol_list2` to identify and remove duplicates based on
    a specified RMSD cutoff value. It performs an initial filtering within `mol_list1`, removing molecules
    with RMSD values below `cut_off_val` between them. Then, it checks the remaining unique molecules in
    `mol_list1` against `mol_list2` to ensure no duplicates are present between the two lists.

    Parameters:
        mol_list1 (List[Tuple[str, Molecule]]): A list of tuples, where each tuple contains an identifier
            (str) and a Molecule object to be filtered based on binding site uniqueness.
        mol_list2 (List[Tuple[str, Molecule]]): A secondary list of tuples, where each tuple contains an
            identifier (str) and a Molecule object used to verify binding site uniqueness with `mol_list1`.
        cut_off_val (float): The RMSD threshold value; molecules with RMSD less than this value are considered duplicates.
        rmsd_symm (bool): If True, considers molecular symmetry when computing RMSD, comparing both regular
            and mirror-image RMSD values.

    Returns:
        List[Molecule]: A list of unique Molecule objects from `mol_list1`, with duplicates filtered out based on RMSD.

    Logs:
        Logs the total number of duplicate binding sites removed.

    """
    logger.info("\nStarting filtering procedure:")

    logger.info("Comparing within structures found in this round:")
    # Initial filtering within mol_list1
    to_remove_tmp = set()  
    for i in range(len(mol_list1)):
        for j in range(i + 1, len(mol_list1)):
            rmsd_val, rmsd_val_mirror = compute_rmsd(
                mol_list1[i][1], mol_list1[j][1], rmsd_symm
            )
            if (rmsd_val < cut_off_val) or (rmsd_val_mirror < cut_off_val):
                min_rmsd = min(rmsd_val, rmsd_val_mirror)
                logger.info(f"Duplicate found: {mol_list1[i][0]} vs {mol_list1[j][0]}, RMSD: {min_rmsd}")
                to_remove_tmp.add(mol_list1[j][0])  # Store identifiers for uniqueness

    # Creating unique_tmp list by excluding duplicates found in to_remove_tmp
    unique_tmp = [mol for mol in mol_list1 if mol[0] not in to_remove_tmp]

    logger.info("Comparing with strucutres already present in the Optimization Dataset")
    # Further filtering against mol_list2
    to_remove_final = set()  
    for unique_mol in unique_tmp:
        for mol in mol_list2:
            rmsd_val, rmsd_val_mirror = compute_rmsd(
                unique_mol[1], mol[1], rmsd_symm
            )
            if (rmsd_val < cut_off_val) or (rmsd_val_mirror < cut_off_val):
                min_rmsd = min(rmsd_val, rmsd_val_mirror)
                logger.info(f"Duplicate found: {unique_mol[0]} vs. {mol[0]}, RMSD: {min_rmsd}")
                to_remove_final.add(unique_mol[0])

    # Calculate final unique molecules, excluding those in to_remove_final
    total_removed = len(to_remove_tmp) + len(to_remove_final)
    unique_final = [mol for mol in unique_tmp if mol[0] not in to_remove_final]
    logger.info(f"{total_removed} duplicate  and {len(unique_final)} unique binding sites.")
    return unique_final


def run_sampling(
    method: str,
    basis: str,
    program: str,
    tag: str,
    kw_id: str,
    sampling_opt_dset: OptimizationDataset,
    refinement_opt_dset: OptimizationDataset,
    opt_lot: str,
    rmsd_symm: bool,
    store_initial: bool,
    rmsd_val: float,
    target_mol: Molecule,
    cluster: Molecule,
    debug_path: Path,
    client: FractalClient,
    sampling_shell: float,
    sampling_condition: str,
    logger: logging.Logger,
):

    # Defining initial variables
    FREQUENCY = 120
    ATOMS_PER_CLUSTER_MOL = 3
    binding_site_num = 0
    n_smpl_mol = 0

    # Calculating max number of sampling structure for each shell and the number of shells
    max_structures = int(max(3, (len(cluster.symbols) / ATOMS_PER_CLUSTER_MOL) // 3))
    shell_list = generate_shell_list(sampling_shell, sampling_condition)

    logger.info(
        f"Entering the sampling prodecure, will generate a total of {max_structures} structures for each shell. {len(shell_list)} shells will be sampled."
    )

    # cleaning up the terachem method
    if program == "terachem":
        if len(method.split("-")) == 2:
            method = method.split("-")[0]

    spec = {
        "name": opt_lot,
        "description": "Geometric Optimziation ",
        "optimization_spec": {"program": "geometric", "keywords": {"maxiter": 125}},
        "qc_spec": {
            "driver": "gradient",
            "method": method,
            "basis": basis,
            "keywords": kw_id,
            "program": program,
        },
    }

    sampling_opt_dset.add_specification(**spec, overwrite=True)
    logger.info(
        "Adding the specification {} to the {} OptimizationData set.".format(
            spec["name"], sampling_opt_dset.name
        )
    )

    for shell in shell_list:
        logger.info(f"\nStarting sampling within a shell of {shell} Angstrom")

        # Define lists
        entry_name_list = []

        # Generate entry names
        entry_base_name = refinement_opt_dset.name
        for i in range(max_structures):
            n_smpl_mol += 1
            entry_name = entry_base_name + "_" + str(n_smpl_mol).zfill(4)
            entry_name_list.append(entry_name)

        # Check if entries already exist in refinement OptimizationDataset
        total_existing_entries = list(sampling_opt_dset.data.records)

        shell_new_entries = [
            e for e in entry_name_list if e not in total_existing_entries
        ]
        shell_old_entries = [e for e in entry_name_list if e in total_existing_entries]

        logger.info(
            "Number of existing entries: {}   {}".format(
                len(shell_old_entries), " ".join(shell_old_entries)
            )
        )
        logger.info(
            "Number of new entries: {}   {}".format(
                len(shell_new_entries), " ".join(shell_new_entries)
            )
        )

        # Update counter
        n_smpl_mol -= len(shell_new_entries)

        # Get IDs of the optimization
        pid_list = get_job_ids(sampling_opt_dset, shell_old_entries, opt_lot)

        if pid_list:
            logger.debug(f"Procedure IDs of the optimization are: {pid_list}")

        # Wait for old jobs to finish
        wait_for_completion(client, pid_list, FREQUENCY, logger)

        # If there are no new entries to be generated continue
        if not shell_new_entries:
            logger.info(
                "All entries for this shell  already exits, will proceed to the next shell"
            )
            continue

        # The number of initial structures to be generated
        max_structures = len(shell_new_entries)

        # Get the initial molecules from sampler function
        molecules, debug_mol = mol_sample(
            cluster,
            target_mol,
            sampling_shell=shell,
            max_structures=max_structures,
            debug=True,
        )

        # Add the molecules to the sampling dataset
        logger.info(
            f"Adding entries for {len(molecules)} new molecules to the {sampling_opt_dset.name} OptimizationDataset "
        )
        for i, m in enumerate(molecules):
            n_smpl_mol += 1
            try:
                sampling_opt_dset.add_entry(shell_new_entries[i], m, save=True)
                new_mols = True
            except KeyError as e:
                logger.info(e)

        if store_initial:
            logger.info(
                f"Initial structure set for visualization will be saved in {str(debug_path)}"
            )
            filename = f"{debug_path}_{round(shell, 2):.2f}".replace(".", "")+'.mol'
            debug_mol.to_file(filename, "xyz")

        # Send sampling computation
        comp_rec = sampling_opt_dset.compute(opt_lot, tag=tag)
        logger.info(f"{comp_rec} sampling optimization procedures were submitted!")

        # Get IDs of the optimization
        pid_list = get_job_ids(sampling_opt_dset, entry_name_list, opt_lot)
        logger.info(
            "Procedure IDs of the optimization are: {}".format(" ".join(pid_list))
        )

        # Checks if no more jobs are running
        wait_for_completion(client, pid_list, FREQUENCY, logger)

        # Gets the optimized molecules for the completed jobs
        opt_molecules_new = get_opt_molecules(
            sampling_opt_dset, entry_name_list, opt_lot, status="COMPLETE"
        )
        opt_mol_num = len(opt_molecules_new)
        logger.debug(
            f"{opt_mol_num} COMPLETED molecules in {sampling_opt_dset.name} for this round, {opt_mol_num - len(molecules)} molecules ended in ERROR."
        )

        # Getting existing molecule ids for RMSD filtering
        opt_molecules = []
        for optentry in refinement_opt_dset.data.records.items():
            mol_id = optentry[1].initial_molecule
            entry_name = optentry[0]
            mol_obj = client.query_molecules(mol_id)[0]
            opt_molecules.append((entry_name, mol_obj))

        # Filter the molecules by RMSD
        logger.info(
            f"Filtering {opt_mol_num} new molecules against existing {len(opt_molecules)} molecules  using an RMSD criteria of {rmsd_val}"
        )
        unique_mols = filter_binding_sites(
            opt_molecules_new, opt_molecules, cut_off_val=rmsd_val, rmsd_symm=rmsd_symm, logger=logger
        )

        # Add the molecules to the refinement OptimizationDataset
        for mol_info in unique_mols:
            entry_name, mol_obj = mol_info
            try:
                refinement_opt_dset.add_entry(entry_name, mol_obj, save=True)
            except KeyError as e:
                logger.info(f"{e} in {refinement_opt_dset.name}")

        new_mols = len(unique_mols)
        binding_site_num += new_mols

        logger.info(
            f"A total of {new_mols} unique binding sites were found after filtering for shell {shell} Angstrom. \nAdding this new binding sites to {refinement_opt_dset.name} for refined optimization."
        )

    total_bind_sites = len(refinement_opt_dset.data.records)
    logger.info(
        f"Finished sampling the cluster. Found {binding_site_num} unique binding sites. Total binding sites: {total_bind_sites}"
    )
    return None
