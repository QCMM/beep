"""
QCFractal Adapter — single module encapsulating ALL QCFractal server I/O.

Updated for QCPortal v0.63+ API.

Every function that touches the QCFractal server lives here. The rest of the
BEEP codebase (core/, workflows/) calls these functions instead of importing
qcportal directly.

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

from qcportal import PortalClient, PortalRequestError
from qcportal.record_models import RecordStatusEnum, PriorityEnum
from qcportal.singlepoint.record_models import QCSpecification, SinglepointDriver
from qcportal.optimization.record_models import OptimizationSpecification
from qcportal.optimization.dataset_models import OptimizationDataset
from qcportal.singlepoint.dataset_models import SinglepointDataset
from qcportal.reaction.dataset_models import ReactionDataset
from qcportal.reaction.record_models import ReactionSpecification, ReactionKeywords
from qcelemental.models.molecule import Molecule
from pydantic import ValidationError

from ..core.logging_utils import log_formatted_list, padded_log
from ..core.stoichiometry import be_stoichiometry
from ..core.errors import DatasetNotFound, LevelOfTheoryNotFound

# Backward-compatible aliases
FractalClient = PortalClient
Dataset = SinglepointDataset

# Re-export types so that workflows can import them from the adapter
# instead of reaching into qcportal internals directly.
__all__ = [
    "FractalClient", "PortalClient", "Dataset", "SinglepointDataset",
    "OptimizationDataset", "ReactionDataset", "Molecule",
    "RecordStatusEnum",
    "is_complete", "is_incomplete", "is_error", "status_label",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

STOICH_TYPES = ("default", "be_nocp", "ie", "de")


_COLLECTION_TYPE_MAP = {
    "OptimizationDataset": "optimization",
    "ReactionDataset": "reaction",
    "Dataset": "singlepoint",
}


def _stoich_dataset_name(base_name: str, stoich_type: str) -> str:
    """Build the dataset name for a given stoichiometry type."""
    return f"{base_name}_{stoich_type}"


def _resolve_dataset_type(collection_type):
    """Convert old collection_type string or class to v0.63 dataset_type string."""
    if isinstance(collection_type, str):
        return _COLLECTION_TYPE_MAP.get(collection_type, collection_type.lower())
    name = getattr(collection_type, "__name__", str(collection_type))
    return _COLLECTION_TYPE_MAP.get(name, name.lower())


def is_complete(status):
    """Check if a record status represents completion."""
    return status == RecordStatusEnum.complete


def is_incomplete(status):
    """Check if a record status represents a running/waiting state."""
    return status in (RecordStatusEnum.running, RecordStatusEnum.waiting)


def is_error(status):
    """Check if a record status represents an error."""
    return status == RecordStatusEnum.error


def status_label(status):
    """Map v0.63 RecordStatusEnum to old-style uppercase string."""
    if is_complete(status):
        return "COMPLETE"
    if is_incomplete(status):
        return "INCOMPLETE"
    if is_error(status):
        return "ERROR"
    return str(status.value).upper()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(address: str = "localhost:7777", username: str = None,
            password: str = None, verify: bool = False) -> PortalClient:
    """Create and return a PortalClient connection."""
    return PortalClient(
        address=address, verify=verify,
        username=username, password=password,
        show_motd=False,
    )


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def get_collection(client: PortalClient, collection_type: str, name: str):
    """Retrieve a dataset by type and name.

    Raises KeyError if the dataset does not exist (preserves v0.15 behavior).
    """
    ds_type = _resolve_dataset_type(collection_type)
    try:
        return client.get_dataset(ds_type, name)
    except (KeyError, PortalRequestError) as e:
        raise KeyError(
            f"Dataset '{name}' of type '{ds_type}' not found"
        ) from e


def get_or_create_opt_dataset(client: PortalClient, name: str):
    """Get an existing OptimizationDataset or create a new one."""
    try:
        return client.get_dataset("optimization", name)
    except (KeyError, PortalRequestError):
        return client.add_dataset("optimization", name)


def create_reaction_dataset(client: PortalClient, name: str,
                            program: str = "psi4"):
    """Delete existing and create a fresh ReactionDataset."""
    try:
        ds = client.get_dataset("reaction", name)
        client.delete_dataset(ds.id, delete_records=False)
    except (KeyError, PortalRequestError):
        pass
    return client.add_dataset("reaction", name)


def delete_collection(client: PortalClient, collection_type: str,
                      name: str) -> None:
    """Delete a dataset, ignoring errors if it doesn't exist."""
    try:
        ds_type = _resolve_dataset_type(collection_type)
        ds = client.get_dataset(ds_type, name)
        client.delete_dataset(ds.id, delete_records=False)
    except (KeyError, PortalRequestError):
        pass


def check_collection_exists(client: PortalClient, collection_type: str,
                            name: str) -> bool:
    """Check whether a dataset exists on the server."""
    ds_type = _resolve_dataset_type(collection_type)
    try:
        client.get_dataset(ds_type, name)
        return True
    except (KeyError, PortalRequestError):
        return False


def check_collection_existence(client: PortalClient, *collections,
                                collection_type: str = "OptimizationDataset") -> None:
    """Validate that all named datasets exist. Raises DatasetNotFound if any is missing."""
    logger = logging.getLogger("beep")
    for collection in collections:
        if not check_collection_exists(client, collection_type, collection):
            raise DatasetNotFound(
                f"Collection {collection} does not exist. "
                "Please create it first. Exiting..."
            )
        ds_type = _resolve_dataset_type(collection_type)
        logger.info(f"The {ds_type} dataset named {collection} exists \u2714")


def get_or_create_collection(client: PortalClient, name: str,
                             collection_type=OptimizationDataset):
    """Get or create a dataset of any type."""
    logger = logging.getLogger("beep")
    ds_type = _resolve_dataset_type(collection_type)
    type_name = (collection_type.__name__
                 if hasattr(collection_type, "__name__")
                 else str(collection_type))
    try:
        ds = client.get_dataset(ds_type, name)
        logger.info(
            f"Collection of type {type_name} with name {name} "
            "already exists. \u2714\n"
        )
    except (KeyError, PortalRequestError):
        ds = client.add_dataset(ds_type, name)
        logger.info(f"Creating new {type_name}: {name}.\n")
    return ds


def check_optimized_molecule(ds, opt_lot: str, mol_names) -> None:
    """Check that all named molecules have COMPLETE optimization records."""
    if isinstance(mol_names, str):
        mol_names = [mol_names]
    for mol in list(mol_names):
        record = ds.get_record(mol, opt_lot)
        if record is None:
            raise LevelOfTheoryNotFound(
                f"{opt_lot} level of theory for {mol} or the entry itself "
                f"does not exist in {ds.name} dataset. "
                "Add the molecule and optimize it first\n"
            )
        if is_incomplete(record.status):
            raise ValueError(
                f" Optimization has status {record.status} restart it or wait"
            )
        elif is_error(record.status):
            raise ValueError(
                f" Optimization has status {record.status} restart it or wait"
            )


def get_molecular_multiplicity(client: PortalClient, dataset,
                               molecule_name: str) -> int:
    """Get spin multiplicity from the initial molecule of a dataset entry."""
    entry = dataset.get_entry(molecule_name.lower())
    if entry is None:
        entry = dataset.get_entry(molecule_name)
    if entry is None:
        raise KeyError(f"Entry '{molecule_name}' not found in dataset")
    return entry.initial_molecule.molecular_multiplicity


def get_xyz(client: PortalClient, dataset: str, mol_name: str,
            level_theory: str,
            collection_type: str = "OptimizationDataset") -> str:
    """Fetch optimized geometry as XYZ string."""
    ds_opt = get_collection(client, collection_type, dataset)
    record = fetch_opt_record(ds_opt, mol_name, level_theory)
    mol = record.final_molecule
    geom = mol.to_string(dtype="xyz")
    xyz_list = geom.splitlines()[2:]
    return "\n".join(xyz_list)


def wait_for_dataset_completion(client: PortalClient, ds_opt, opt_lot: str,
                                 logger: logging.Logger,
                                 wait_interval: int = 600) -> None:
    """Poll a dataset until all records for a given LOT are COMPLETE."""
    while True:
        incomplete = []
        n_complete = 0
        n_error = 0
        for entry_name in ds_opt.entry_names:
            try:
                record = ds_opt.get_record(entry_name, opt_lot)
                if record is not None:
                    if is_incomplete(record.status):
                        incomplete.append(entry_name)
                    elif is_error(record.status):
                        n_error += 1
                        logger.info(
                            f"Warning: Entry '{entry_name}' finished with ERROR"
                        )
                    elif is_complete(record.status):
                        n_complete += 1
            except (KeyError, TypeError):
                continue

        n_total = n_complete + len(incomplete) + n_error
        if not incomplete:
            logger.info(
                f"Dataset '{ds_opt.name}': {n_complete}/{n_total} COMPLETE"
                + (f", {n_error} ERROR" if n_error else "")
                + ". All done."
            )
            return

        logger.info(
            f"Dataset '{ds_opt.name}': {n_complete}/{n_total} COMPLETE, "
            f"{len(incomplete)} incomplete"
            + (f", {n_error} ERROR" if n_error else "")
            + f". Waiting {wait_interval}s..."
        )
        time.sleep(wait_interval)
        # Re-fetch to get updated statuses
        ds_opt = get_collection(client, "OptimizationDataset", ds_opt.name)


# ---------------------------------------------------------------------------
# Molecule queries
# ---------------------------------------------------------------------------

def fetch_molecules(client: PortalClient, mol_ids) -> List[Molecule]:
    """Query molecules by ID(s). Always returns a list."""
    if not isinstance(mol_ids, (list, tuple)):
        mol_ids = [mol_ids]
    result = client.get_molecules(mol_ids)
    if not isinstance(result, list):
        return [result]
    return result


def fetch_opt_record(ds_opt, entry_name: str, opt_lot: str):
    """Get the optimization record for an entry.

    Raises KeyError if the entry or record does not exist.
    """
    record = ds_opt.get_record(entry_name, opt_lot)
    if record is None:
        raise KeyError(
            f"No record for entry '{entry_name}' with specification '{opt_lot}'"
        )
    return record


def fetch_final_molecule(ds_opt, entry_name: str, opt_lot: str) -> Molecule:
    """Get the final optimized molecule from an optimization record."""
    return fetch_opt_record(ds_opt, entry_name, opt_lot).final_molecule


def fetch_initial_molecule(ds_opt, entry_name: str, opt_lot: str) -> Molecule:
    """Get the initial molecule from an optimization record."""
    return fetch_opt_record(ds_opt, entry_name, opt_lot).initial_molecule


def fetch_opt_molecules(ds_opt, entry_list: List[str], opt_lot: str,
                        status: str = "COMPLETE") -> List[Tuple[str, Molecule]]:
    """
    Retrieve final molecules from an optimization dataset for entries with
    the specified status.
    """
    target_status = RecordStatusEnum(status.lower())
    mol_list = []
    for n in entry_list:
        record = ds_opt.get_record(n, opt_lot)
        if record is not None and record.status == target_status:
            mol_list.append((n, record.final_molecule))
    return mol_list


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def add_opt_specification(ds_opt, spec_dict: dict,
                          overwrite: bool = True) -> None:
    """Add an optimization specification to a dataset.

    Accepts old-style spec dicts with 'name', 'description',
    'optimization_spec', and 'qc_spec' keys and translates to v0.63
    OptimizationSpecification objects.
    """
    name = spec_dict.get("name", "default")
    description = spec_dict.get("description", "")
    qc_spec_dict = spec_dict.get("qc_spec", {})
    opt_spec_dict = spec_dict.get("optimization_spec", {})

    # In v0.63, keywords are dicts, not server-stored IDs
    qc_keywords = qc_spec_dict.get("keywords", {})
    if not isinstance(qc_keywords, dict):
        qc_keywords = {}

    qc_spec = QCSpecification(
        program=qc_spec_dict.get("program", "psi4"),
        driver=SinglepointDriver.deferred,
        method=qc_spec_dict.get("method", ""),
        basis=qc_spec_dict.get("basis"),
        keywords=qc_keywords,
    )

    opt_keywords = opt_spec_dict.get("keywords", {})
    if not isinstance(opt_keywords, dict):
        opt_keywords = {}

    opt_spec = OptimizationSpecification(
        program=opt_spec_dict.get("program", "geometric"),
        qc_specification=qc_spec,
        keywords=opt_keywords,
    )

    if overwrite:
        try:
            ds_opt.delete_specification(name, delete_records=False)
        except Exception:
            pass

    ds_opt.add_specification(name, opt_spec, description=description)


def add_opt_entry(ds_opt, name: str, molecule: Molecule,
                  save: bool = True) -> None:
    """Add an entry to an optimization dataset.

    The ``save`` parameter is accepted for backward compatibility but
    is ignored — v0.63 entries are saved immediately via the API.
    """
    ds_opt.add_entry(name=name, initial_molecule=molecule)


def submit_optimizations(ds_opt, opt_lot: str, tag: str, subset=None):
    """Submit optimization computations."""
    entry_names = list(subset) if subset else None
    return ds_opt.submit(
        entry_names=entry_names,
        specification_names=[opt_lot],
        compute_tag=tag,
    )


def submit_energies(client: PortalClient, rdset_base_name: str,
                    method: str, basis: str, program: str,
                    stoich: str, tag: str, keywords=None):
    """Submit energy computations to a stoichiometry-specific ReactionDataset."""
    ds_name = _stoich_dataset_name(rdset_base_name, stoich)
    ds = client.get_dataset("reaction", ds_name)
    spec_name = f"{method}_{basis}"

    kw_dict = keywords if isinstance(keywords, dict) else {}
    qc_spec = QCSpecification(
        program=program,
        driver=SinglepointDriver.energy,
        method=method,
        basis=basis,
        keywords=kw_dict,
    )
    rxn_spec = ReactionSpecification(
        program="reaction",
        singlepoint_specification=qc_spec,
        keywords=ReactionKeywords(),
    )
    try:
        ds.add_specification(spec_name, rxn_spec)
    except Exception:
        pass  # specification already exists

    return ds.submit(
        specification_names=[spec_name],
        compute_tag=tag,
    )


def submit_hessians(client: PortalClient, program: str, method: str,
                    basis: str, mol_ids: list, keywords, tag: str):
    """Submit hessian computations via client.add_singlepoints."""
    kw_dict = keywords if isinstance(keywords, dict) else {}
    return client.add_singlepoints(
        molecules=mol_ids,
        program=program,
        driver="hessian",
        method=method,
        basis=basis,
        keywords=kw_dict,
        compute_tag=tag,
    )


def add_reaction(client: PortalClient, rdset_base_name: str,
                 name: str, stoichiometry: dict) -> None:
    """Add reaction entries across stoichiometry-specific datasets.

    ``stoichiometry`` is a dict mapping stoich type names to lists of
    ``(Molecule, coefficient)`` tuples (as returned by ``be_stoichiometry``).
    Each stoich type is stored in its own ReactionDataset named
    ``{rdset_base_name}_{stoich_type}``.
    """
    for stoich_type, mol_coeff_list in stoichiometry.items():
        ds_name = _stoich_dataset_name(rdset_base_name, stoich_type)
        ds = client.get_dataset("reaction", ds_name)
        # be_stoichiometry returns (Molecule, coeff); v0.63 wants (coeff, Molecule)
        stoichiometries = [(coeff, mol) for mol, coeff in mol_coeff_list]
        ds.add_entry(name=name, stoichiometries=stoichiometries)


def add_keywords(client: PortalClient, keyword_set) -> Any:
    """In v0.63 keywords are inline in specifications — returns the dict."""
    if isinstance(keyword_set, dict):
        return keyword_set
    return getattr(keyword_set, "values", keyword_set)


def create_keyword_set(values: dict) -> dict:
    """In v0.63, keywords are plain dicts. Returns values directly."""
    return values


def query_keywords(client: PortalClient):
    """Not available in v0.63 — keywords are inline in specifications."""
    raise NotImplementedError(
        "query_keywords is not available in QCPortal v0.63+. "
        "Keywords are now inline in specifications."
    )


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

def get_job_ids(ds_opt, entry_list: List[str], opt_lot: str) -> List[int]:
    """Get record IDs for optimization entries."""
    pid_list = []
    for n in entry_list:
        record = ds_opt.get_record(n, opt_lot)
        if record is not None:
            pid_list.append(record.id)
    return pid_list


def check_for_completion(client: PortalClient, pid: List[int],
                         frequency: int = 60) -> Tuple[bool, Counter]:
    """
    Check job completion status. Returns (all_complete, status_counts).

    Status counts use old-style uppercase strings for backward compatibility.
    """
    status_labels = []
    if pid:
        records = client.get_records(pid, missing_ok=True)
        if not isinstance(records, list):
            records = [records]
        for r in records:
            if r is not None:
                status_labels.append(status_label(r.status))
    status_counts = Counter(status_labels)
    all_jobs_complete = "INCOMPLETE" not in status_labels
    return all_jobs_complete, status_counts


def wait_for_completion(client: PortalClient, pid_list: List[int],
                        frequency: int, logger: logging.Logger) -> None:
    """Poll until all jobs are complete."""
    jobs_complete = False
    logger.info("Checking for job completion....")
    while not jobs_complete:
        jobs_complete, counts = check_for_completion(client, pid_list, frequency)
        status_str = " ".join(
            [f"{s}: {count}, " for s, count in counts.items()]
        )
        logger.info("The status of the Optimization jobs: " + status_str)
        if not jobs_complete:
            time.sleep(frequency)


def check_jobs_status(client: PortalClient, job_ids: List[int],
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
            records = client.get_records(chunk, missing_ok=True)
            if not isinstance(records, list):
                records = [records]

            for job in records:
                if job is not None:
                    label = status_label(job.status)
                    if label in status_counts:
                        status_counts[label] += 1
                    else:
                        logger.info(
                            f"Job ID {job.id}: Unknown status - {job.status}"
                        )
                else:
                    logger.info("Job not found in the database")

        logger.info(
            f"Job Status Summary: {status_counts['INCOMPLETE']} INCOMPLETE, "
            f"{status_counts['COMPLETE']} COMPLETE, "
            f"{status_counts['ERROR']} ERROR\n"
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

def fetch_reaction_entries(client: PortalClient, rdset_base_name: str,
                           stoich: str = "be_nocp"):
    """Get entries and their molecule IDs from a stoich-specific ReactionDataset.

    Returns a list of dicts with keys ``name``, ``stoichiometry``, ``molecule``
    (molecule ID), ``coefficient``, matching the old DataFrame format used by
    ``compute_hessian``.
    """
    import pandas as pd
    ds_name = _stoich_dataset_name(rdset_base_name, stoich)
    ds = client.get_dataset("reaction", ds_name)
    rows = []
    for entry in ds.iterate_entries():
        for s in entry.stoichiometries:
            rows.append({
                "name": entry.name,
                "stoichiometry": stoich,
                "molecule": s.molecule.id,
                "coefficient": s.coefficient,
            })
    return pd.DataFrame(rows)


def fetch_reaction_values(client: PortalClient, rdset_base_name: str,
                          stoich: str = "default", spec_name: str = None):
    """Get computed reaction energies from a stoich-specific ReactionDataset.

    Returns a DataFrame with entry names as index and specification names
    as columns, containing ``total_energy`` values (in Hartree).
    """
    import pandas as pd
    import qcelemental as qcel

    ds_name = _stoich_dataset_name(rdset_base_name, stoich)
    ds = client.get_dataset("reaction", ds_name)

    spec_names = [spec_name] if spec_name else ds.specification_names
    data = {}
    for sname in spec_names:
        col_label = sname.replace("_", "/")
        energies = {}
        for entry_name, sn, record in ds.iterate_records(
            specification_names=[sname],
            status=RecordStatusEnum.complete,
        ):
            if record is not None and record.total_energy is not None:
                energies[entry_name] = (
                    record.total_energy * qcel.constants.hartree2kcalmol
                )
        data[col_label] = energies

    return pd.DataFrame(data)


def query_results(client: PortalClient, driver: str, molecule,
                  method: str, basis: str):
    """Query singlepoint result records from the server."""
    return list(client.query_singlepoints(
        driver=driver,
        molecule_id=molecule if isinstance(molecule, int) else None,
        method=method,
        basis=basis,
    ))


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

    for index in ds_opt.entry_names:
        try:
            record = ds_opt.get_record(index, opt_lot)
            if record is not None and is_complete(record.status):
                molecule_records[index] = record.final_molecule
        except (ValidationError, TypeError) as e:
            logger.warning(
                f"Error retrieving record {index}, "
                "Optimization finished with ERROR"
            )
            continue

    molecule_keys: List[str] = list(molecule_records.keys())
    count = 0
    for i in range(len(molecule_keys)):
        for j in range(i + 1, len(molecule_keys)):
            count += 1
            mol1 = molecule_records[molecule_keys[i]]
            mol2 = molecule_records[molecule_keys[j]]
            rmsd = mol1.align(mol2, atoms_map=True)[1]["rmsd"]
            logger.debug(
                f"RMSD between {molecule_keys[i]} and "
                f"{molecule_keys[j]}: {rmsd}"
            )
            if rmsd < 0.25 and rmsd != 0.0:
                if molecule_keys[j] not in molecules_to_delete:
                    molecules_to_delete.append(molecule_keys[j])

    logger.info(f"List of molecules to delete: {molecules_to_delete}")
    unique_molecules_to_delete: List[str] = list(set(molecules_to_delete))
    for molecule_key in unique_molecules_to_delete:
        del molecule_records[molecule_key]

    logger.info(
        f"Remaining molecules after filtering: "
        f"{list(molecule_records.keys())}"
    )
    return molecule_records


# ---------------------------------------------------------------------------
# Composite operations
# ---------------------------------------------------------------------------

def create_or_load_reaction_dataset(
    client: PortalClient,
    rdset_name: str,
    opt_lot: str,
    smol_mol: Molecule,
    cluster_mol: Molecule,
    ds_opt,
    opt_stru: Dict[str, object],
    logger: logging.Logger,
) -> str:
    """
    Create stoichiometry-specific ReactionDatasets and populate with
    benchmark structures.

    Creates one dataset per stoichiometry type (default, be_nocp, ie, de),
    named ``{rdset_name}_{stoich_type}``.

    Returns the base dataset name (without stoich suffix).
    """
    # Create (or recreate) one dataset per stoich type
    for stoich_type in STOICH_TYPES:
        ds_name = _stoich_dataset_name(rdset_name, stoich_type)
        try:
            ds = client.get_dataset("reaction", ds_name)
            client.delete_dataset(ds.id, delete_records=False)
        except (KeyError, PortalRequestError):
            pass
        client.add_dataset("reaction", ds_name)

    n_entries = 0
    padded_log(
        logger,
        f"Populating the dataframe with {len(opt_stru.keys())} new entries",
        padding_char="*", total_length=60,
    )

    for st in opt_stru.keys():
        logger.info(f"Processing structure: {st}")
        rr = ds_opt.get_record(st, opt_lot)

        if rr is None or is_error(rr.status):
            logger.warning(
                f"WARNING: Optimization of {st} with {opt_lot} finished "
                "with error. Will skip this structure."
            )
            continue

        struct_mol = rr.final_molecule
        logger.info(f"Generating BE stoichiometry for {st}")
        be_stoich = be_stoichiometry(smol_mol, cluster_mol, struct_mol, logger)

        n_entries += 1
        try:
            add_reaction(client, rdset_name, st, be_stoich)
            logger.info(f"Successfully added {st} to the datasets.\n")
        except (KeyError, Exception) as e:
            logger.warning(f"Failed to add {st}. Skipping entry. {e}")
            continue

    logger.info(f"Created a total of {n_entries} entries in {rdset_name}.\n")
    return rdset_name


def compute_be_dft_energies(
    client: PortalClient,
    rdset_base_name: str,
    all_dft: List[str],
    tag: str,
    program: str,
    logger: logging.Logger,
    keyword=None,
) -> List[int]:
    """
    Submit DFT energy computations for BE calculations.

    Submits across all stoichiometry-specific datasets. Returns a list of
    record IDs for monitoring.
    """
    logger.info(
        f"Computing energies for the following stoichiometries: "
        f"{' '.join(STOICH_TYPES)} (default = be)\n"
    )

    log_formatted_list(
        logger, all_dft,
        "Sending DFT energy computations for the following Levels of theory:",
        max_rows=1,
    )

    logger.info(f"\nSending DFT computations with tag: {tag}\n")

    all_submitted = 0
    all_existing = 0

    for i, lot in enumerate(all_dft):
        method, basis = lot.split("_")
        logger.info(f"Processing method: {method}, basis: {basis}")

        lot_submitted = 0
        lot_existing = 0

        for stoich in STOICH_TYPES:
            result = submit_energies(
                client, rdset_base_name,
                method=method, basis=basis, program=program,
                stoich=stoich, tag=tag, keywords=keyword,
            )
            lot_submitted += result.n_inserted
            lot_existing += result.n_existing

        all_submitted += lot_submitted
        all_existing += lot_existing
        logger.info(
            f"{lot}: Existing {lot_existing}  Submitted {lot_submitted}"
        )

    logger.info(
        f"\nSubmitted a total of {all_submitted} DFT computations. "
        f"{all_existing} are already computed."
    )

    # Collect record IDs for monitoring
    record_ids = []
    for stoich in STOICH_TYPES:
        ds_name = _stoich_dataset_name(rdset_base_name, stoich)
        ds = client.get_dataset("reaction", ds_name)
        for _, _, record in ds.iterate_records(
            status=[RecordStatusEnum.complete, RecordStatusEnum.running,
                    RecordStatusEnum.waiting, RecordStatusEnum.error],
        ):
            if record is not None:
                record_ids.append(record.id)

    return record_ids


def compute_hessian(
    client: PortalClient,
    ds_be_name: str,
    opt_lot: str,
    mult: int,
    hess_tag: str,
    logger: logging.Logger,
    program: str = "psi4",
) -> List[int]:
    """
    Compute hessian for all molecules in the be_nocp ReactionDataset.
    """
    ds_name = _stoich_dataset_name(ds_be_name, "be_nocp")
    try:
        ds_nocp = client.get_dataset("reaction", ds_name)
    except (KeyError, PortalRequestError):
        logger.info(
            f"\nWARNING: Reaction dataset {ds_name} does not exist.\n"
        )
        return []

    padded_log(
        logger, f"Computing Hessians for {ds_be_name}",
        padding_char="*", total_length=60,
    )

    method, basis = opt_lot.split("_")

    # Collect unique molecule IDs from be_nocp entries
    mol_ids = set()
    for entry in ds_nocp.iterate_entries():
        for s in entry.stoichiometries:
            mol_ids.add(s.molecule.id)

    # Remove single atoms
    u_mols = list(mol_ids)
    mol_list = client.get_molecules(u_mols)
    if not isinstance(mol_list, list):
        mol_list = [mol_list]
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
    kw = {"function_kwargs": {"dertype": 1}}
    if mult == 2:
        kw["reference"] = "uks"

    logger.info(f"\nComputing Hessian at {method}/{basis} level of theory")
    logger.info(f"Using keywords: {kw}")

    meta, record_ids = client.add_singlepoints(
        molecules=u_mols,
        program=program,
        driver="hessian",
        method=method,
        basis=basis,
        keywords=kw,
        compute_tag=hess_tag,
    )

    logger.info(
        f"\nExisting: {meta.n_existing} Submitted: {meta.n_inserted}"
    )
    logger.info(
        f"\nSubmitted a total of {meta.n_inserted} computations. "
        f"{meta.n_existing} are already computed."
    )
    return record_ids


# ---------------------------------------------------------------------------
# ZPVE (QCF-coupled version)
# ---------------------------------------------------------------------------

def get_zpve_mol(client: PortalClient, mol, lot_opt: str,
                 on_imaginary: str = "return"):
    """
    Compute vibrational analysis for a molecule and handle imaginary frequencies.
    Fetches hessian data from QCFractal.
    """
    from ..core.zpve import _vibanal_wfn

    logger = logging.getLogger("beep")
    mol_list = fetch_molecules(client, mol)
    mol_obj = mol_list[0]
    num_atm = len(mol_obj.symbols)

    if num_atm == 1:
        logger.info(
            f"Molecule {mol} is an atom, will return 0.0 for ZPVE"
        )
        return 0.0, True

    mol_form = mol_obj.dict()["identifiers"]["molecular_formula"]
    lot_parts = lot_opt.split("_", 1)
    method = lot_parts[0]
    basis = lot_parts[1] if len(lot_parts) == 2 else None
    if method[0] == "U":
        method = method[1:]

    try:
        results = list(client.query_singlepoints(
            driver=SinglepointDriver.hessian,
            molecule_id=mol,
            method=method,
            basis=basis,
        ))
        result = results[0]
        logger.info(
            f"Molecule {mol} with molecular formula {mol_form} "
            "has a Hessian calculation."
        )
    except (IndexError, StopIteration):
        logger.info(
            f"Molecule {mol} with molecular formula {mol_form} "
            "has not finished computing."
        )
        return None, True

    hess = result.return_result
    # Energy may be in extras.qcvars or in properties
    result_dict = result.dict()
    qcvars = result_dict.get("extras", {}).get("qcvars", {})
    energy = qcvars.get("CURRENT ENERGY")
    if energy is None:
        energy = (result.properties or {}).get("current energy")

    result_mol = result.molecule

    vib, therm = _vibanal_wfn(
        hess=hess, molecule=result_mol, energy=energy
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
    kw_id,
    sampling_opt_dset,
    refinement_opt_dset,
    opt_lot: str,
    rmsd_symm: bool,
    store_initial: bool,
    rmsd_val: float,
    target_mol: Molecule,
    cluster: Molecule,
    debug_path,
    client: PortalClient,
    sampling_shell: float,
    sampling_condition: str,
    logger: logging.Logger,
):
    """
    Run the full sampling loop: generate structures, optimize, filter by RMSD.
    """
    from ..core.sampling import generate_shell_list, filter_binding_sites
    from ..core.molecule_sampler import random_molecule_sampler as mol_sample

    FREQUENCY = 120
    ATOMS_PER_CLUSTER_MOL = 3
    binding_site_num = 0
    n_smpl_mol = 0

    max_structures = int(
        max(3, (len(cluster.symbols) / ATOMS_PER_CLUSTER_MOL) // 3)
    )
    shell_list = generate_shell_list(sampling_shell, sampling_condition)

    logger.info(
        f"Entering the sampling procedure, will generate a total of "
        f"{max_structures} structures for each shell. "
        f"{len(shell_list)} shells will be sampled."
    )

    if program == "terachem":
        if len(method.split("-")) == 2:
            method = method.split("-")[0]

    # Build v0.63 optimization specification
    qc_keywords = kw_id if isinstance(kw_id, dict) else {}
    qc_spec = QCSpecification(
        program=program,
        driver=SinglepointDriver.deferred,
        method=method,
        basis=basis,
        keywords=qc_keywords,
    )
    opt_spec = OptimizationSpecification(
        program="geometric",
        qc_specification=qc_spec,
        keywords={"maxiter": 125},
    )

    try:
        sampling_opt_dset.delete_specification(opt_lot, delete_records=False)
    except Exception:
        pass
    sampling_opt_dset.add_specification(
        opt_lot, opt_spec, description="Geometric Optimization"
    )
    logger.info(
        "Adding the specification {} to the {} OptimizationDataset.".format(
            opt_lot, sampling_opt_dset.name
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

        total_existing_entries = list(sampling_opt_dset.entry_names)

        shell_new_entries = [
            e for e in entry_name_list if e not in total_existing_entries
        ]
        shell_old_entries = [
            e for e in entry_name_list if e in total_existing_entries
        ]

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
            logger.debug(
                f"Procedure IDs of the optimization are: {pid_list}"
            )

        wait_for_completion(client, pid_list, FREQUENCY, logger)

        if not shell_new_entries:
            logger.info(
                "All entries for this shell already exist, "
                "will proceed to the next shell"
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
        new_mols = False
        for i, m in enumerate(molecules):
            n_smpl_mol += 1
            try:
                sampling_opt_dset.add_entry(
                    name=shell_new_entries[i], initial_molecule=m,
                )
                new_mols = True
            except KeyError as e:
                logger.info(e)

        if store_initial:
            logger.info(
                f"Initial structure set for visualization will be saved "
                f"in {str(debug_path)}"
            )
            filename = (
                f"{debug_path}_{round(shell, 2):.2f}".replace(".", "")
                + ".mol"
            )
            debug_mol.to_file(filename, "xyz")

        comp_rec = sampling_opt_dset.submit(
            specification_names=[opt_lot], compute_tag=tag,
        )
        logger.info(
            f"{comp_rec.n_inserted} new, {comp_rec.n_existing} existing "
            "sampling optimization procedures."
        )

        pid_list = get_job_ids(sampling_opt_dset, entry_name_list, opt_lot)
        logger.info(
            "Procedure IDs of the optimization are: {}".format(
                " ".join(str(p) for p in pid_list)
            )
        )

        wait_for_completion(client, pid_list, FREQUENCY, logger)

        opt_molecules_new = fetch_opt_molecules(
            sampling_opt_dset, entry_name_list, opt_lot, status="COMPLETE"
        )
        opt_mol_num = len(opt_molecules_new)
        logger.debug(
            f"{opt_mol_num} COMPLETED molecules in "
            f"{sampling_opt_dset.name} for this round, "
            f"{opt_mol_num - len(molecules)} molecules ended in ERROR."
        )

        # Get existing molecules from refinement dataset
        opt_molecules = []
        for entry in refinement_opt_dset.iterate_entries():
            opt_molecules.append((entry.name, entry.initial_molecule))

        logger.info(
            f"Filtering {opt_mol_num} new molecules against existing "
            f"{len(opt_molecules)} molecules using an RMSD criteria "
            f"of {rmsd_val}"
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
                refinement_opt_dset.add_entry(
                    name=entry_name, initial_molecule=mol_obj,
                )
            except KeyError as e:
                logger.info(f"{e} in {refinement_opt_dset.name}")

        new_mols_count = len(unique_mols)
        binding_site_num += new_mols_count

        logger.info(
            f"A total of {new_mols_count} unique binding sites were found "
            f"after filtering for shell {shell} Angstrom. \n"
            f"Adding this new binding sites to {refinement_opt_dset.name} "
            "for refined optimization."
        )

    total_bind_sites = len(refinement_opt_dset.entry_names)
    logger.info(
        f"Finished sampling the cluster. Found {binding_site_num} unique "
        f"binding sites. Total binding sites: {total_bind_sites}"
    )
    return None
