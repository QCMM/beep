import sys
import time
import logging
import numpy as np
from pathlib import Path
from qcfractal.interface.collections.optimization_dataset import OptimizationDataset
from qcelemental.models.molecule import Molecule
from qcfractal.interface.client import FractalClient
from typing import List, Tuple, Optional
from collections import Counter, defaultdict
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


def compute_rmsd_conditional(m1, m2, rmsd_symm: bool, cutoff: float):
    r = m1.align(m2, atoms_map=True)[1]["rmsd"]
    if rmsd_symm and r >= cutoff:
        rm = m1.align(m2, atoms_map=True, run_mirror=True)[1]["rmsd"]
    else:
        rm = 10.0
    return r, rm

def ligand_dm_ok(m1: Molecule, m2: Molecule, ligand_size: int, tau: float = 1e-3) -> bool:
    """Cheap, rotation/translation-invariant gate based on the ligand distance matrix."""
    X1 = np.asarray(m1.geometry)[-ligand_size:]
    X2 = np.asarray(m2.geometry)[-ligand_size:]
    d1 = X1[:, None, :] - X1[None, :, :]
    d2 = X2[:, None, :] - X2[None, :, :]
    D1 = np.linalg.norm(d1, axis=2)
    D2 = np.linalg.norm(d2, axis=2)
    # Frobenius norm tolerance in Å (tune tau as needed)
    return np.linalg.norm(D1 - D2) <= tau

def _key_for_grid(mol: Molecule, ligand_size: int, grid: float) -> Tuple[int, int, int]:
    lig = np.asarray(mol.geometry)[-ligand_size:]
    com = lig.mean(axis=0)
    return tuple((com / grid).astype(int))

def _neighbor_keys(key: Tuple[int, int, int], radius: int):
    ix, iy, iz = key
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                yield (ix + dx, iy + dy, iz + dz)

# ---------- main filter ----------

def filter_binding_sites(
    mol_list1: List[Tuple[str, Molecule]],
    mol_list2: List[Tuple[str, Molecule]],
    cut_off_val: float,
    rmsd_symm: bool,
    ligand_size: int,
    logger: logging.Logger,
    grid: float = 0.5,
    nb_radius: int = 3,
    dm_tau: float = 1e-3,
) -> List[Tuple[str, Molecule]]:
    """
    Filters duplicates using:
      1) Ligand distance-matrix gate (invariant, very cheap).
      2) Spatial hashing by ligand COM (grid=0.5 Å) + neighbor buckets (radius=3).
      3) Conditional mirror RMSD only when needed.

    Returns list of (name, Molecule) from mol_list1 that are unique.
    """
    logger.info("\nStarting filtering procedure:")
    logger.info("Comparing within structures found in this round:")

    # Precompute keys for list1
    l1 = [(name, mol, _key_for_grid(mol, ligand_size, grid)) for (name, mol) in mol_list1]

    # 1) Within-round dedup (search across neighbor voxels)
    to_remove_tmp = set()
    for i in range(len(l1)):
        ni, mi, ki = l1[i]
        if ni in to_remove_tmp:
            continue
        for j in range(i + 1, len(l1)):
            nj, mj, kj = l1[j]
            if nj in to_remove_tmp:
                continue
            # compare only if keys are within the neighbor radius (Chebyshev metric)
            if (abs(ki[0] - kj[0]) > nb_radius or
                abs(ki[1] - kj[1]) > nb_radius or
                abs(ki[2] - kj[2]) > nb_radius):
                continue
            # invariant ligand gate first
            if not ligand_dm_ok(mi, mj, ligand_size, tau=dm_tau):
                continue
            # then the real RMSD
            r, rm = compute_rmsd_conditional(mi, mj, rmsd_symm, cut_off_val)
            if min(r, rm) < cut_off_val:
                logger.info(f"Duplicate found: {ni} vs {nj}, RMSD: {min(r, rm):.3f}")
                to_remove_tmp.add(nj)

    unique_tmp = [(name, mol) for (name, mol, _) in l1 if name not in to_remove_tmp]

    # 2) Against reference set (neighbor buckets + invariant gate)
    logger.info("Comparing with structures already present in the Optimization Dataset")

    # Bucket refs
    buckets2 = defaultdict(list)
    for rname, rmol in mol_list2:
        k = _key_for_grid(rmol, ligand_size, grid)
        buckets2[k].append((rname, rmol))

    to_remove_final = set()
    for name, mol in unique_tmp:
        k = _key_for_grid(mol, ligand_size, grid)
        # gather candidates from neighbor voxels
        candidates = []
        for nk in _neighbor_keys(k, nb_radius):
            if nk in buckets2:
                candidates.extend(buckets2[nk])

        # neighbor pass
        dropped = False
        for rname, rmol in candidates:
            if not ligand_dm_ok(mol, rmol, ligand_size, tau=dm_tau):
                continue
            r, rm = compute_rmsd_conditional(mol, rmol, rmsd_symm, cut_off_val)
            if min(r, rm) < cut_off_val:
                logger.info(f"Duplicate found: {name} vs. {rname}, RMSD: {min(r, rm):.3f}")
                to_remove_final.add(name)
                dropped = True
                break
        if dropped:
            continue

    total_removed = len(to_remove_tmp) + len(to_remove_final)
    unique_final = [pair for pair in unique_tmp if pair[0] not in to_remove_final]
    logger.info(f"{total_removed} duplicates removed. {len(unique_final)} unique binding sites remain.")
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
        mol_size = len(target_mol.symbols)
        unique_mols = filter_binding_sites(
            opt_molecules_new, opt_molecules, cut_off_val=rmsd_val, rmsd_symm=rmsd_symm, ligand_size=mol_size, logger=logger,
            grid=0.5, nb_radius=3, dm_tau=1e-3 
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
