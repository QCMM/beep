"""
QCFractal Adapter — single module encapsulating ALL QCFractal server I/O.

Every function that touches the QCFractal server lives here. The rest of the
BEEP codebase (core/, workflows/) calls these functions instead of importing
qcfractal.interface directly.

Functions are organized by category:
  - Connection
  - Collection management
  - Molecule queries
  - Job submission
  - Job monitoring
  - Data retrieval
  - RMSD filter (server-coupled version)
  - ZPVE (server-coupled version)
"""
import time
import logging
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter

import qcportal as ptl
from qcportal.client import FractalClient
from qcportal.collections import Dataset, OptimizationDataset, ReactionDataset
from qcelemental.models.molecule import Molecule
from pydantic import ValidationError

from ..core.logging_utils import log_formatted_list, padded_log
from ..core.stoichiometry import be_stoichiometry


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(address: str = "localhost:7777", username: str = None,
            password: str = None, verify: bool = False) -> FractalClient:
    """Create and return a FractalClient connection."""
    return ptl.FractalClient(
        address=address, verify=verify,
        username=username, password=password,
    )


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def get_collection(client: FractalClient, collection_type: str, name: str):
    """Retrieve a collection by type and name."""
    return client.get_collection(collection_type, name)


def get_or_create_opt_dataset(client: FractalClient, name: str) -> OptimizationDataset:
    """Get an existing OptimizationDataset or create a new one."""
    try:
        ds_opt = client.get_collection("OptimizationDataset", name)
    except KeyError:
        ds_opt = ptl.collections.OptimizationDataset(name, client=client)
        ds_opt.save()
        ds_opt = client.get_collection("OptimizationDataset", name)
    return ds_opt


def create_reaction_dataset(client: FractalClient, name: str,
                            program: str = "psi4") -> ReactionDataset:
    """Delete existing and create a fresh ReactionDataset."""
    try:
        client.delete_collection("ReactionDataset", name)
    except KeyError:
        pass
    ds_be = ReactionDataset(name, ds_type="rxn", client=client,
                            default_program=program)
    ds_be.save()
    return client.get_collection("ReactionDataset", name)


def delete_collection(client: FractalClient, collection_type: str, name: str) -> None:
    """Delete a collection, ignoring KeyError if it doesn't exist."""
    try:
        client.delete_collection(collection_type, name)
    except KeyError:
        pass


def check_collection_exists(client: FractalClient, collection_type: str, name: str) -> bool:
    """Check whether a collection exists on the server."""
    try:
        client.get_collection(collection_type, name)
        return True
    except KeyError:
        return False


# ---------------------------------------------------------------------------
# Molecule queries
# ---------------------------------------------------------------------------

def fetch_molecules(client: FractalClient, mol_ids) -> List[Molecule]:
    """Query molecules by ID(s)."""
    return client.query_molecules(mol_ids)


def fetch_opt_record(ds_opt: OptimizationDataset, entry_name: str, opt_lot: str):
    """Get the optimization record for an entry."""
    return ds_opt.get_record(entry_name, opt_lot)


def fetch_final_molecule(ds_opt: OptimizationDataset, entry_name: str,
                         opt_lot: str) -> Molecule:
    """Get the final optimized molecule from an optimization record."""
    return ds_opt.get_record(entry_name, opt_lot).get_final_molecule()


def fetch_initial_molecule(ds_opt: OptimizationDataset, entry_name: str,
                           opt_lot: str) -> Molecule:
    """Get the initial molecule from an optimization record."""
    return ds_opt.get_record(entry_name, opt_lot).get_initial_molecule()


def fetch_opt_molecules(ds_opt: OptimizationDataset, entry_list: List[str],
                        opt_lot: str,
                        status: str = "COMPLETE") -> List[Tuple[str, Molecule]]:
    """
    Retrieve final molecules from an optimization dataset for entries with
    the specified status.
    """
    mol_list = []
    for n in entry_list:
        record = ds_opt.get_record(n, opt_lot)
        if record.status == status:
            mol_list.append((n, record.get_final_molecule()))
    return mol_list


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def add_opt_specification(ds_opt: OptimizationDataset, spec_dict: dict,
                          overwrite: bool = True) -> None:
    """Add an optimization specification to a dataset."""
    ds_opt.add_specification(**spec_dict, overwrite=overwrite)


def add_opt_entry(ds_opt: OptimizationDataset, name: str,
                  molecule: Molecule, save: bool = True) -> None:
    """Add an entry to an optimization dataset."""
    ds_opt.add_entry(name, molecule, save=save)


def submit_optimizations(ds_opt: OptimizationDataset, opt_lot: str,
                         tag: str, subset=None):
    """Submit optimization computations."""
    return ds_opt.compute(opt_lot, tag=tag, subset=subset)


def submit_energies(ds_be: ReactionDataset, method: str, basis: str,
                    program: str, stoich: str, tag: str,
                    keywords=None):
    """Submit energy computations to a ReactionDataset."""
    return ds_be.compute(
        method=method, basis=basis, program=program,
        stoich=stoich, tag=tag, keywords=keywords,
    )


def submit_hessians(client: FractalClient, program: str, method: str,
                    basis: str, mol_ids: list, kw_id, tag: str):
    """Submit hessian computations via client.add_compute."""
    return client.add_compute(program, method, basis, "hessian",
                              kw_id, mol_ids, tag=tag)


def add_reaction(ds_be: ReactionDataset, name: str, stoichiometry: dict) -> None:
    """Add a reaction entry to a ReactionDataset."""
    ds_be.add_rxn(name, stoichiometry)


def add_keywords(client: FractalClient, keyword_set) -> str:
    """Add a keyword set and return its ID."""
    return client.add_keywords([keyword_set])[0]


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

def get_job_ids(ds_opt: OptimizationDataset, entry_list: List[str],
                opt_lot: str) -> List[int]:
    """Get procedure IDs for optimization entries."""
    pid_list = []
    for n in entry_list:
        opt_rec = ds_opt.get_record(n, opt_lot)
        pid_list.append(opt_rec.id)
    return pid_list


def check_for_completion(client: FractalClient, pid: List[str],
                         frequency: int = 60) -> Tuple[bool, Counter]:
    """
    Check job completion status. Returns (all_complete, status_counts).
    """
    status = []
    if pid:
        status = [r.status for r in client.query_procedures(pid)]
    status_counts = Counter(status)
    all_jobs_complete = "INCOMPLETE" not in status
    return all_jobs_complete, status_counts


def wait_for_completion(client: FractalClient, pid_list: List[str],
                        frequency: int, logger: logging.Logger) -> None:
    """Poll until all jobs are complete."""
    jobs_complete = False
    logger.info("Checking for job completion....")
    while not jobs_complete:
        jobs_complete, counts = check_for_completion(client, pid_list, frequency)
        status_str = " ".join([f"{s}: {count}, " for s, count in counts.items()])
        logger.info("The status of the Optimization jobs: " + status_str)
        if not jobs_complete:
            time.sleep(frequency)


def check_jobs_status(client: FractalClient, job_ids: List[str],
                      logger: logging.Logger, wait_interval: int = 600,
                      print_job_ids: bool = False) -> None:
    """
    Continuously monitor and report computation status, processing in chunks.
    """
    all_complete = False
    chunk_size = 1000

    while not all_complete:
        status_counts = {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}

        for i in range(0, len(job_ids), chunk_size):
            chunk = job_ids[i:i + chunk_size]
            if print_job_ids:
                logger.info(f"List with job_ids: {chunk}")
            job_stats = client.query_procedures(chunk)

            for job in job_stats:
                if job:
                    status = job.status.upper()
                    if status in status_counts:
                        status_counts[status] += 1
                    else:
                        logger.info(f"Job ID {job.id}: Unknown status - {status}")
                else:
                    logger.info(f"Job ID {job.id}: Not found in the database")

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
            time.sleep(wait_interval)


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------

def fetch_reaction_entries(ds_be: ReactionDataset):
    """Get all entries from a ReactionDataset as a DataFrame."""
    return ds_be.get_entries()


def fetch_reaction_values(ds_be: ReactionDataset, stoich: str = None,
                          method: str = None):
    """Get computed values from a ReactionDataset."""
    kwargs = {}
    if stoich:
        kwargs["stoich"] = stoich
    if method:
        kwargs["method"] = method
    return ds_be.get_values(**kwargs)


def query_results(client: FractalClient, driver: str, molecule,
                  method: str, basis: str):
    """Query result records from the server."""
    return client.query_results(driver=driver, molecule=molecule,
                                method=method, basis=basis)


# ---------------------------------------------------------------------------
# RMSD filter (QCF-coupled version that fetches from server)
# ---------------------------------------------------------------------------

def rmsd_filter_from_dataset(ds_opt, opt_lot: str,
                             logger: logging.Logger) -> Dict[str, Molecule]:
    """
    Filter molecules from an OptimizationDataset by RMSD < 0.25 A.
    Fetches final molecules from the server for comparison.
    """
    logger.info("Starting rmsd_filter")
    molecules_to_delete: List[str] = []
    molecule_records: Dict[str, Molecule] = {}

    for index in ds_opt.df.index:
        try:
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
            rmsd = mol1.align(mol2, atoms_map=True)[1]["rmsd"]
            logger.debug(f"RMSD between {molecule_keys[i]} and {molecule_keys[j]}: {rmsd}")
            if rmsd < 0.25 and rmsd != 0.0:
                if molecule_keys[j] not in molecules_to_delete:
                    molecules_to_delete.append(molecule_keys[j])

    logger.info(f"List of molecules to delete: {molecules_to_delete}")
    unique_molecules_to_delete: List[str] = list(set(molecules_to_delete))
    for molecule_key in unique_molecules_to_delete:
        del molecule_records[molecule_key]

    logger.info(f"Remaining molecules after filtering: {list(molecule_records.keys())}")
    return molecule_records


# ---------------------------------------------------------------------------
# Composite operations (extracted from binding_energy_compute.py)
# ---------------------------------------------------------------------------

def create_or_load_reaction_dataset(
    client: FractalClient,
    rdset_name: str,
    opt_lot: str,
    smol_mol: Molecule,
    cluster_mol: Molecule,
    ds_opt: Dataset,
    opt_stru: Dict[str, object],
    logger: logging.Logger,
) -> ReactionDataset:
    """
    Create or update a ReactionDataset with benchmark structures.
    Extracted from binding_energy_compute.py.
    """
    try:
        client.delete_collection("ReactionDataset", rdset_name)
    except KeyError:
        pass

    ds_be = ReactionDataset(rdset_name, ds_type="rxn", client=client,
                            default_program="psi4")
    ds_be.save()
    ds_be = client.get_collection("ReactionDataset", rdset_name)

    n_entries = 0
    padded_log(logger, f"Populating the dataframe with {len(opt_stru.keys())} new entries",
               padding_char="*", total_length=60)

    for st in opt_stru.keys():
        logger.info(f"Processing structure: {st}")
        rr = ds_opt.get_record(st, opt_lot)

        if rr.status == "ERROR":
            logger.warning(
                f"WARNING: Optimization of {st} with {opt_lot} finished with error. "
                "Will skip this structure."
            )
            continue

        struct_mol = rr.get_final_molecule()
        logger.info(f"Generating BE stoichiometry for {st}")
        be_stoich = be_stoichiometry(smol_mol, cluster_mol, struct_mol, logger)

        rds_entry = f"{st}"
        n_entries += 1
        try:
            ds_be.add_rxn(rds_entry, be_stoich)
            logger.info(f"Successfully added {st} to the dataset.\n")
        except KeyError:
            logger.warning(f"Failed to add {st}. Skipping entry.")
            continue

    ds_be.save()
    logger.info(f"Created a total of {n_entries} entries in {rdset_name}.\n")
    return ds_be


def compute_be_dft_energies(
    ds_be,
    all_dft: List[str],
    tag: str,
    program: str,
    logger: logging.Logger,
    keyword: str = None,
) -> List[str]:
    """
    Submit DFT energy computations for BE calculations.
    Extracted from binding_energy_compute.py.
    """
    stoich_list = ["default", "de", "ie", "be_nocp"]
    logger.info(
        f"Computing energies for the following stoichiometries: "
        f"{' '.join(stoich_list)} (default = be)\n"
    )

    log_formatted_list(
        logger, all_dft,
        "Sending DFT energy computations for the following Levels of theory:",
        max_rows=1,
    )

    c_list_sub = []
    c_list_exis = []

    logger.info(f"\nSending DFT computations with tag: {tag}\n")

    for i, lot in enumerate(all_dft):
        method, basis = lot.split("_")
        logger.info(f"Processing method: {method}, basis: {basis}")

        c_per_lot_sub = []
        c_per_lot_exis = []

        for stoich in stoich_list:
            c = ds_be.compute(
                method=method, basis=basis, program=program,
                stoich=stoich, tag=tag, keywords=keyword,
            )
            c_list_sub.extend(list(c)[1][1])
            c_per_lot_sub.extend(list(c)[1][1])
            c_list_exis.extend(list(c)[0][1])
            c_per_lot_exis.extend(list(c)[0][1])

        logger.info(
            f"{lot}: Existing {len(c_per_lot_exis)}  Submitted {len(c_per_lot_sub)}"
        )

    logger.info(
        f"\nSubmitted a total of {len(c_list_sub)} DFT computations. "
        f"{len(c_list_exis)} are already computed."
    )
    return c_list_sub + c_list_exis


def compute_hessian(
    client: FractalClient,
    ds_be_name: str,
    opt_lot: str,
    mult: int,
    hess_tag: str,
    logger: logging.Logger,
    program: str = "psi4",
) -> List[str]:
    """
    Compute hessian for all molecules in a ReactionDataset.
    Extracted from binding_energy_compute.py.
    """
    try:
        ds_be = client.get_collection("ReactionDataset", ds_be_name)
    except KeyError:
        logger.info(f"\nWARNING: Reaction database {ds_be_name} does not exist.\n")
        return []

    padded_log(logger, f"Computing Hessians for {ds_be.name}",
               padding_char="*", total_length=60)

    method, basis = opt_lot.split("_")
    df_all = ds_be.get_entries()
    mols = df_all[df_all["stoichiometry"] == "be_nocp"]["molecule"]
    u_mols = list(set(mols))

    mol_list = client.query_molecules(u_mols)
    for m in mol_list:
        if len(m.symbols) == 1:
            logger.info(f"\nAtom id {m.id} removed")
            u_mols.remove(m.id)

    log_formatted_list(
        logger, u_mols,
        "Sending Hessian computations for the following molecules:",
        max_rows=5,
    )

    logger.info(f"\nWill compute Hessian at {method}/{basis} level of theory")
    if mult == 2:
        kw = ptl.models.KeywordSet(
            **{"values": {"function_kwargs": {"dertype": 1}, "reference": "uks"}}
        )
    else:
        kw = ptl.models.KeywordSet(
            **{"values": {"function_kwargs": {"dertype": 1}}}
        )

    logger.info(f"\nComputing Hessian at {method}/{basis} level of theory")
    logger.info(f"Using keywords: {kw.values}")

    kw_id = client.add_keywords([kw])[0]
    c = client.add_compute(program, method, basis, "hessian", kw_id, u_mols,
                           tag=hess_tag)

    c_list_sub = list(c)[1][1]
    c_list_exis = list(c)[0][1]

    logger.info(f"\nExisting: {len(c_list_exis)} Submitted: {len(c_list_sub)}")
    logger.info(
        f"\nSubmitted a total of {len(c_list_sub)} computations. "
        f"{len(c_list_exis)} are already computed."
    )
    return c_list_sub + c_list_exis


# ---------------------------------------------------------------------------
# ZPVE (QCF-coupled version)
# ---------------------------------------------------------------------------

def get_zpve_mol(client: FractalClient, mol: int, lot_opt: str,
                 on_imaginary: str = "return"):
    """
    Compute vibrational analysis for a molecule and handle imaginary frequencies.
    Extracted from zpve.py — this function fetches data from QCFractal.
    """
    from ..core.zpve import _vibanal_wfn

    logger = logging.getLogger("beep")
    mol_obj = client.query_molecules(mol)[0]
    num_atm = len(mol_obj.symbols)

    if num_atm == 1:
        logger.info(f"Molecule {mol} is an atom, will retrun 0.0 for ZPVE")
        return 0.0, True

    mol_form = mol_obj.dict()["identifiers"]["molecular_formula"]
    method = lot_opt.split("_")[0]
    basis = lot_opt.split("_")[1]
    if method[0] == "U":
        method = method[1:]

    try:
        result = client.query_results(
            driver="hessian", molecule=mol, method=method, basis=basis
        )[0]
        logger.info(
            f"Molecule {mol} with molecular formula {mol_form} "
            "has a Hessian calculation."
        )
    except IndexError:
        logger.info(
            f"Molecule {mol} with molecular formula {mol_form} "
            "has not finished computing."
        )
        return None, True

    hess = result.dict()["return_result"]
    energy = result.dict()["extras"]["qcvars"]["CURRENT ENERGY"]
    vib, therm = _vibanal_wfn(
        hess=hess, molecule=result.get_molecule(), energy=energy
    )

    imag_freqs = [num for num in vib["omega"].data if abs(num.imag) > 1]
    if imag_freqs:
        if on_imaginary == "raise":
            import numpy as np
            np.savetxt(f"{mol}_hess.dat", hess, fmt="%.18e")
            raise ValueError(
                f"There are imaginary frequencies: {imag_freqs}. "
                f"You need to reoptimize {mol}."
            )
        elif on_imaginary == "return":
            logger.info(
                f"There are imaginary frequencies: {imag_freqs}, "
                "proceed with caution."
            )
            return therm["ZPE_vib"].data, False
        else:
            raise ValueError(
                f"Invalid option for on_imaginary: {on_imaginary}"
            )

    return therm["ZPE_vib"].data, True


# ---------------------------------------------------------------------------
# Sampling (QCF-coupled version)
# ---------------------------------------------------------------------------

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
    debug_path,
    client: FractalClient,
    sampling_shell: float,
    sampling_condition: str,
    logger: logging.Logger,
):
    """
    Run the full sampling loop: generate structures, optimize, filter by RMSD.
    Extracted from beep/sampling.py — this function interacts heavily with QCFractal.
    """
    from ..core.sampling import generate_shell_list, filter_binding_sites
    from ..core.molecule_sampler import random_molecule_sampler as mol_sample

    FREQUENCY = 120
    ATOMS_PER_CLUSTER_MOL = 3
    binding_site_num = 0
    n_smpl_mol = 0

    max_structures = int(max(3, (len(cluster.symbols) / ATOMS_PER_CLUSTER_MOL) // 3))
    shell_list = generate_shell_list(sampling_shell, sampling_condition)

    logger.info(
        f"Entering the sampling prodecure, will generate a total of {max_structures} "
        f"structures for each shell. {len(shell_list)} shells will be sampled."
    )

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

        entry_name_list = []
        entry_base_name = refinement_opt_dset.name
        for i in range(max_structures):
            n_smpl_mol += 1
            entry_name = entry_base_name + "_" + str(n_smpl_mol).zfill(4)
            entry_name_list.append(entry_name)

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

        n_smpl_mol -= len(shell_new_entries)

        pid_list = get_job_ids(sampling_opt_dset, shell_old_entries, opt_lot)

        if pid_list:
            logger.debug(f"Procedure IDs of the optimization are: {pid_list}")

        wait_for_completion(client, pid_list, FREQUENCY, logger)

        if not shell_new_entries:
            logger.info(
                "All entries for this shell  already exits, will proceed to the next shell"
            )
            continue

        max_structures = len(shell_new_entries)

        molecules, debug_mol = mol_sample(
            cluster,
            target_mol,
            sampling_shell=shell,
            max_structures=max_structures,
            debug=True,
        )

        logger.info(
            f"Adding entries for {len(molecules)} new molecules to the "
            f"{sampling_opt_dset.name} OptimizationDataset "
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
            filename = f"{debug_path}_{round(shell, 2):.2f}".replace(".", "") + '.mol'
            debug_mol.to_file(filename, "xyz")

        comp_rec = sampling_opt_dset.compute(opt_lot, tag=tag)
        logger.info(f"{comp_rec} sampling optimization procedures were submitted!")

        pid_list = get_job_ids(sampling_opt_dset, entry_name_list, opt_lot)
        logger.info(
            "Procedure IDs of the optimization are: {}".format(" ".join(pid_list))
        )

        wait_for_completion(client, pid_list, FREQUENCY, logger)

        opt_molecules_new = fetch_opt_molecules(
            sampling_opt_dset, entry_name_list, opt_lot, status="COMPLETE"
        )
        opt_mol_num = len(opt_molecules_new)
        logger.debug(
            f"{opt_mol_num} COMPLETED molecules in {sampling_opt_dset.name} for this round, "
            f"{opt_mol_num - len(molecules)} molecules ended in ERROR."
        )

        opt_molecules = []
        for optentry in refinement_opt_dset.data.records.items():
            mol_id = optentry[1].initial_molecule
            entry_name = optentry[0]
            mol_obj = client.query_molecules(mol_id)[0]
            opt_molecules.append((entry_name, mol_obj))

        logger.info(
            f"Filtering {opt_mol_num} new molecules against existing {len(opt_molecules)} "
            f"molecules  using an RMSD criteria of {rmsd_val}"
        )
        mol_size = len(target_mol.symbols)
        unique_mols = filter_binding_sites(
            opt_molecules_new, opt_molecules, cut_off_val=rmsd_val,
            rmsd_symm=rmsd_symm, ligand_size=mol_size, logger=logger,
            grid=0.5, nb_radius=4, dm_tau=1e-3,
        )

        for mol_info in unique_mols:
            entry_name, mol_obj = mol_info
            try:
                refinement_opt_dset.add_entry(entry_name, mol_obj, save=True)
            except KeyError as e:
                logger.info(f"{e} in {refinement_opt_dset.name}")

        new_mols = len(unique_mols)
        binding_site_num += new_mols

        logger.info(
            f"A total of {new_mols} unique binding sites were found after filtering "
            f"for shell {shell} Angstrom. \nAdding this new binding sites to "
            f"{refinement_opt_dset.name} for refined optimization."
        )

    total_bind_sites = len(refinement_opt_dset.data.records)
    logger.info(
        f"Finished sampling the cluster. Found {binding_site_num} unique binding sites. "
        f"Total binding sites: {total_bind_sites}"
    )
    return None
