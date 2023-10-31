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
    }

    shell_list = conditions_map.get(condition)
    if shell_list is None:
        raise ValueError("Condition should be one of ['sparse', 'normal', 'fine']")

    return shell_list


# def clean_optdset(ds_opt: OptimizationDataset, spec_name = 'str') -> int:
#    # Get the error entires
#    df = ds_opt.df
#    error_entries = df[df[spec_name].apply(lambda x: x.status == 'ERROR')].index.tolist()


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

    status = [r.status for r in client.query_procedures(pid)]
    status_counts = Counter(status)

    # Check if all jobs have the status "ERROR"
    if len(status) == status_counts["ERROR"]:
        raise RuntimeError(
            "All jobs have the status 'ERROR'. Please delete the OptimizationDataset and check your Specification"
        )

    all_jobs_complete = "INCOMPLETE" not in status

    return all_jobs_complete, status_counts


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
        align_mols_mirror = mol1.align(mol2, run_mirror=True)
        rmsd_val_mirror = align_mols_mirror[1]["rmsd"]
    else:
        align_mols = mol1.align(mol2, atoms_map=True)
        rmsd_val = align_mols[1]["rmsd"]
    return rmsd_val, rmsd_val_mirror


def filter_binding_sites(
    mol_list1: List[Tuple[str, Molecule]],
    mol_list2: List[Tuple[str, Molecule]],
    cut_off_val: float,
    rmsd_symm: bool,
) -> List[Molecule]:
    logger = logging.getLogger("beep_logger")
    to_remove_tmp = []
    for i in range(len(mol_list1)):
        for j in range(i + 1, len(mol_list1)):
            rmsd_val, rmsd_val_mirro = compute_rmsd(
                mol_list1[i][1], mol_list1[j][1], rmsd_symm
            )
            if rmsd_val < cut_off_val or rmsd_val_mirror < cut_off_val:
                to_remove.append(mol_list1[j])
    print(to_remove_tmp, mol_list1)
    unique_tmp = [mol for mol in mol_list1 if mol not in to_remove_tmp]

    to_remove_final = []
    for i in range(len(mol_list2)):
        for j in range(len(unique_tmp)):
            rmsd_val, rmsd_val_mirro = compute_rmsd(
                mol_list1[i][1], mol_list1[j][1], rmsd_symm
            )
            if rmsd_val < cut_off_val or rmsd_val_mirror < cut_off_val:
                to_remove_final.append(unique_tmp[j])
    total_removed = len(to_remove_tmp) + len(to_remove_final)
    logger.debug(f"{total_removed} are duplicate binding sites")
    unique_final = [mol for mol in unique_tmp if mol not in to_remove_final]
    return unique_final


def sampling(
    method: str,
    basis: str,
    program: str,
    tag: str,
    kw_id: str,
    sampling_opt_dset: OptimizationDataset,
    refinement_opt_dset: OptimizationDataset,
    opt_lot: str,
    rmsd_symm: bool,
    rmsd_val: float,
    target_mol: Molecule,
    cluster: Molecule,
    o_file: Path,
    client: FractalClient,
    sampling_shell: float,
    sampling_condition: str,
):
    logger = logging.getLogger("beep_logger")

    FREQUENCY = 200

    spec = {
        "name": opt_lot,
        "description": "Geometric Optimziation ",
        "optimization_spec": {"program": "geometric", "keywords": {"maxiter": 100}},
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
        """Adding the specification {} to the {} OptimizationData set.
    """.format(
            spec["name"], sampling_opt_dset.name
        )
    )

    shell_list = generate_shell_list(sampling_shell, sampling_condition)

    binding_site_num = 0
    n_smpl_mol = 0

    # Remove the entries with error from the OptimizationDataset
    # num_removed = clean_optdset(sampling_opt_dset)

    for shell in shell_list:
        logger.info(f"\nStarting sampling within a shell of {shell} Angstrom")

        # Get the molecules from sampler function
        molecules, _ = mol_sample(
            cluster,
            target_mol,
            sampling_shell=shell,
            debug=True,
        )

        # Add the molecules to the sampling dataset
        entry_list = []
        for m in molecules:
            n_smpl_mol += 1
            entry_name = refinement_opt_dset.name + "_" + str(n_smpl_mol).zfill(4)
            entry_list.append(entry_name)
            try:
                sampling_opt_dset.add_entry(entry_name, m, save=True)
            except KeyError as e:
                logger.info(e)

        # Send sampling computation
        comp_rec = sampling_opt_dset.compute(opt_lot, tag=tag)
        logger.info(f"{comp_rec} sampling optimization procedures were submitted!")

        # Get IDs of the optimization
        pid_list = []
        for n in entry_list:
            opt_rec = sampling_opt_dset.get_record(n, opt_lot)
            pid_list.append(opt_rec.id)
        logger.debug(f"Procedure IDs of the optimization are: {pid_list}")

        # Checks if no more jobs are running
        jobs_complete = False
        while not jobs_complete:
            status = []
            jobs_complete, counts = check_for_completion(client, pid_list, FREQUENCY)
            status_str = " ".join([f"{s}: {count}, " for s, count in counts.items()])
            logger.info("The status of the Optimization jobs: " + status_str)
            if not jobs_complete:
                time.sleep(FREQUENCY)

        # Gets the optimized molecules for the completed jobs
        opt_molecules_new = get_opt_molecules(
            sampling_opt_dset, entry_list, opt_lot, status="COMPLETE"
        )
        opt_mol_num = len(opt_molecules_new)
        logger.debug(
            f"{opt_mol_num} COMPLETED molecules in {sampling_opt_dset.name} for this round."
        )

        # Getting existing molecule ids
        ref_ds_opt = refinement_opt_dset
        opt_molecules = []
        for optentry in ref_ds_opt.data.records.items():
            mol_id = optentry[1].initial_molecule
            entry_name = optentry[0]
            mol_obj = client.query_molecules(mol_id)[0]
            opt_molecules.append((entry_name, mol_obj))

        # Filter the molecules by RMSD
        unique_mols = filter_binding_sites(
            opt_molecules_new, opt_molecules, cut_off_val=rmsd_val, rmsd_symm=rmsd_symm
        )
        new_mols = len(unique_mols)
        binding_site_num += new_mols
        logger.info(
            f"A total of {new_mols} unique binding sites were found for shell {shell} Angstrom. "
        )

        # Add the molecules to the refinement OptimizationDataset
        for mol_info in unique_mols:
            entry_name, mol_obj = mol_info
            try:
                ref_ds_opt.add_entry(entry_name, mol_obj, save=True)
            except KeyError as e:
                logger.info(f"{e} in {red_ds_opt.name}")

    total_bind_sites = len(ref_ds_opt.data.records)
    logger.info(
        f"Finished sampling the cluster. Found {binding_site_num} unique binding sites. Total binding sites: {total_bind_sites}"
    )
    return None
