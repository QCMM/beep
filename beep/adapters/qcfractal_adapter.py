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

STOICH_TYPES = ("bsse", "be_nocp", "ie", "de")


_COLLECTION_TYPE_MAP = {
    "OptimizationDataset": "optimization",
    "ReactionDataset": "reaction",
    "Dataset": "singlepoint",
    "SinglepointDataset": "singlepoint",
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
    entry = None
    for name_variant in [molecule_name, molecule_name.upper(), molecule_name.lower()]:
        try:
            entry = dataset.get_entry(name_variant)
            if entry is not None:
                break
        except Exception:
            continue
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

def fetch_atom_molecule(client: PortalClient, atoms_collection: str,
                        atom_name: str) -> Molecule:
    """Fetch an atom's molecule from a SinglepointDataset.

    Atoms (single-atom species) are stored in a dedicated SinglepointDataset
    rather than an OptimizationDataset since they cannot be optimized.
    """
    ds = client.get_dataset("singlepoint", atoms_collection)
    entry = ds.get_entry(atom_name)
    if entry is None:
        raise KeyError(
            f"Atom '{atom_name}' not found in singlepoint dataset "
            f"'{atoms_collection}'"
        )
    return entry.molecule


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
        record = ds_opt.get_record(n, opt_lot, force_refetch=True)
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

    The ``overwrite`` parameter is accepted for backward compatibility but
    is ignored — v0.63 ``add_specification`` is idempotent (silently
    reports existing specs without error).
    """
    name = spec_dict.get("name", "default").lower()
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
                    method: str, basis: Optional[str], program: str,
                    stoich: str, tag: str, keywords=None):
    """Submit energy computations to a stoichiometry-specific ReactionDataset.

    If ``basis`` is ``None`` the spec is named after the method alone (used for
    bare dispersion specs like ``pbe-d3bj`` that have no basis set).
    """
    ds_name = _stoich_dataset_name(rdset_base_name, stoich)
    ds = client.get_dataset("reaction", ds_name)
    spec_name = (f"{method}_{basis}" if basis else method).lower()

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
    ds.add_specification(spec_name, rxn_spec)

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
    """Get record IDs for optimization entries.

    Names not present in the dataset are skipped silently; the server
    raises HTTP 400 on missing entries and we don't want callers to
    crash when they ask for names that were never inserted (e.g. when
    pose generation returns fewer structures than requested).
    """
    known = set(ds_opt.entry_names)
    pid_list = []
    for n in entry_list:
        if n not in known:
            continue
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
                        frequency: int, logger: logging.Logger,
                        max_wait: int = 86400) -> None:
    """Poll until all jobs reach a terminal state (complete or error).

    Parameters
    ----------
    max_wait : int
        Maximum total wait time in seconds (default 24 hours).
        Raises ``TimeoutError`` if exceeded.
    """
    if not pid_list:
        return

    logger.info("Checking for job completion....")
    elapsed = 0
    while True:
        jobs_complete, counts = check_for_completion(client, pid_list, frequency)
        status_str = " ".join(
            f"{s}: {count}" for s, count in counts.items()
        )
        logger.info(f"The status of the Optimization jobs: {status_str}")
        if jobs_complete:
            return
        elapsed += frequency
        if elapsed >= max_wait:
            raise TimeoutError(
                f"Jobs did not complete within {max_wait}s. "
                f"Last status: {status_str}"
            )
        time.sleep(frequency)


def check_jobs_status(client: PortalClient, job_ids: List[int],
                      logger: logging.Logger, wait_interval: int = 600,
                      print_job_ids: bool = False,
                      max_wait: int = 172800,
                      auto_recover_services: bool = True) -> None:
    """
    Continuously monitor and report computation status, processing in chunks.

    Parameters
    ----------
    max_wait : int
        Maximum total wait time in seconds (default 48 hours).
        Raises ``TimeoutError`` if exceeded.
    auto_recover_services : bool
        If True (default), service records (e.g. ReactionRecord) found at
        ERROR status are reset once per ``check_jobs_status`` call. The
        server re-iterates the service: if its child records are now in
        non-error states (e.g. they were reset externally to recover from
        an infrastructure failure), the service transitions to COMPLETE
        on its own; otherwise it returns to ERROR and stays there. Leaf
        records (singlepoints, optimizations) are never auto-reset.
        Set to False to preserve old behavior (ERROR is fully terminal).
    """
    if not job_ids:
        logger.info("No jobs to monitor.")
        return

    chunk_size = 1000
    elapsed = 0
    reset_attempted: set = set()  # service IDs we've already reset this call

    while True:
        status_counts = {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}
        services_to_recover: List[int] = []

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

                    if (auto_recover_services
                            and job.status == RecordStatusEnum.error
                            and getattr(job, "is_service", False)
                            and job.id not in reset_attempted):
                        services_to_recover.append(job.id)
                else:
                    logger.info("Job not found in the database")

        logger.info(
            f"Job Status Summary: {status_counts['INCOMPLETE']} INCOMPLETE, "
            f"{status_counts['COMPLETE']} COMPLETE, "
            f"{status_counts['ERROR']} ERROR\n"
        )

        if services_to_recover:
            logger.info(
                f"Auto-recovering {len(services_to_recover)} errored "
                f"service record(s) — server will re-iterate and transition "
                f"to COMPLETE if children are now OK, else back to ERROR."
            )
            try:
                client.reset_records(services_to_recover)
                reset_attempted.update(services_to_recover)
            except Exception as e:
                logger.warning(
                    f"Auto-recovery reset failed (will retry next cycle): {e}"
                )

        if status_counts["ERROR"] > 0:
            logger.info("Some jobs have ERROR status. Proceed with caution.")

        if status_counts["INCOMPLETE"] == 0 and not services_to_recover:
            logger.info("All jobs are COMPLETE. Continuing with the execution.")
            return

        elapsed += wait_interval
        if elapsed >= max_wait:
            raise TimeoutError(
                f"Jobs did not complete within {max_wait}s. "
                f"Status: {status_counts['INCOMPLETE']} incomplete, "
                f"{status_counts['COMPLETE']} complete, "
                f"{status_counts['ERROR']} error."
            )
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
                          stoich: str = "bsse", spec_name: str = None):
    """Get computed reaction energies from a stoich-specific ReactionDataset.

    Returns a DataFrame with entry names as index and specification names
    as columns, containing ``total_energy`` values converted to kcal/mol.

    Dispersion-corrected DFT is assumed to be stored as a separated pair
    (bare DFT under ``psi4`` + bare dispersion under ``dftd3``/``dftd4``)
    matching the v0.63 migration convention. The bare pieces are summed
    into a composite column (e.g. ``mpwb1k_def2-tzvpd`` + ``mpwb1k-d3bj``
    → ``MPWB1K-D3BJ/DEF2-TZVPD``) for every suffix in
    ``DISPERSION_SUFFIXES``. Integrated specs (method carrying a
    dispersion suffix together with a basis, e.g. ``mpwb1k-d3bj_def2-tzvpd``)
    are skipped with a warning — BEEP no longer supports that layout.
    """
    import pandas as pd
    import qcelemental as qcel

    logger = logging.getLogger("beep")
    ds_name = _stoich_dataset_name(rdset_base_name, stoich)
    ds = client.get_dataset("reaction", ds_name)

    spec_names = [spec_name] if spec_name else ds.specification_names
    data = {}
    for sname in spec_names:
        if "_" in sname and _has_dispersion_suffix(sname.split("_", 1)[0]):
            logger.warning(
                f"Skipping integrated dispersion spec '{sname}' in {ds_name} "
                f"— BEEP expects the separated pair (bare DFT + bare dispersion)."
            )
            continue

        col_label = sname.replace("_", "/").upper()
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

    df = pd.DataFrame(data)

    # Sum bare-DFT + bare-dispersion into composite columns. A bare-dispersion
    # column has no "/" (no basis in its spec name) and ends with one of
    # DISPERSION_SUFFIXES; pair it with a matching "BARE/BASIS" column.
    suffixes_upper = tuple(s.upper() for s in DISPERSION_SUFFIXES)
    disp_cols = [
        c for c in df.columns
        if "/" not in c and any(c.endswith(suf) for suf in suffixes_upper)
    ]
    for disp_col in disp_cols:
        for suffix in suffixes_upper:
            if disp_col.endswith(suffix):
                bare = disp_col[: -len(suffix)]
                break
        dft_matches = [c for c in df.columns if c.startswith(bare + "/")]
        for dft_col in dft_matches:
            basis_part = dft_col.split("/", 1)[1]
            composite_col = f"{disp_col}/{basis_part}"
            df[composite_col] = df[dft_col] + df[disp_col]

    return df


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


# Dispersion suffixes recognized by BEEP's separated-pair dispersion handling.
# Order matters: longer suffixes first so "-d3mbj" matches before "-d3".
# Single source of truth — consumed by compute_be_dft_energies (write side)
# and fetch_reaction_values / extract.py (read side).
DISPERSION_PROGRAMS: Tuple[Tuple[str, str], ...] = (
    ("-d3mbj", "dftd3"),
    ("-d3bj",  "dftd3"),
    ("-d3m",   "dftd3"),
    ("-d4",    "dftd4"),
    ("-d3",    "dftd3"),
)
DISPERSION_SUFFIXES: Tuple[str, ...] = tuple(s for s, _ in DISPERSION_PROGRAMS)


def _split_dispersion(method: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Split ``method`` into (bare_functional, full_dispersion_method, disp_program).

    Returns (method, None, None) when no known dispersion suffix is present
    (plain DFT, HF-3c, WB97X-V / WB97M-V intrinsic VV10, etc.).
    """
    m = method.lower()
    for suffix, program in DISPERSION_PROGRAMS:
        if m.endswith(suffix):
            bare = method[: -len(suffix)]
            return bare, method, program
    return method, None, None


def _has_dispersion_suffix(name: str) -> bool:
    """Return True if ``name`` ends with a known dispersion suffix."""
    m = name.lower()
    return any(m.endswith(s) for s in DISPERSION_SUFFIXES)


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

    For dispersion-corrected functionals (``-d3``, ``-d3bj``, ``-d3m``,
    ``-d3mbj``, ``-d4``) two specs are submitted per level of theory: the
    bare DFT piece on ``program`` (typically psi4) and the bare dispersion
    piece on ``dftd3``/``dftd4`` with ``basis=None``. This matches the
    separated-pair form produced by the v0.15→v0.63 migration script and
    allows ``fetch_reaction_values`` to recombine them into composite
    columns. Non-dispersion methods are submitted as a single integrated
    spec. Submits across all stoichiometry-specific datasets. Returns a
    list of record IDs for monitoring.
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
        bare, disp_method, disp_program = _split_dispersion(method)
        logger.info(f"Processing method: {method}, basis: {basis}")

        lot_submitted = 0
        lot_existing = 0

        for stoich in STOICH_TYPES:
            if disp_method is None:
                # No dispersion suffix — single integrated spec
                result = submit_energies(
                    client, rdset_base_name,
                    method=method, basis=basis, program=program,
                    stoich=stoich, tag=tag, keywords=keyword,
                )
                lot_submitted += result.n_inserted
                lot_existing += result.n_existing
            else:
                # Separated pair: bare DFT + bare dispersion
                dft_result = submit_energies(
                    client, rdset_base_name,
                    method=bare, basis=basis, program=program,
                    stoich=stoich, tag=tag, keywords=keyword,
                )
                disp_result = submit_energies(
                    client, rdset_base_name,
                    method=disp_method, basis=None, program=disp_program,
                    stoich=stoich, tag=tag, keywords=None,
                )
                lot_submitted += dft_result.n_inserted + disp_result.n_inserted
                lot_existing += dft_result.n_existing + disp_result.n_existing

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
        logger, [str(m) for m in u_mols],
        "Sending Hessian computations for the following molecules:",
        max_rows=5,
    )

    logger.info(f"\nWill compute Hessian at {method}/{basis} level of theory")
    kw = {"function_kwargs": {"dertype": 1}}
    if mult != 1:
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
                 on_imaginary: str = "return",
                 imag_threshold: float = 50.0):
    """
    Compute vibrational analysis for a molecule and handle imaginary frequencies.
    Fetches hessian data from QCFractal.

    Parameters
    ----------
    imag_threshold : float
        Imaginary frequencies with magnitude below this value (in cm⁻¹)
        are ignored (dropped from ZPVE sum). Default 50 cm⁻¹.
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

    results = list(client.query_singlepoints(
        driver=SinglepointDriver.hessian,
        molecule_id=mol,
        method=method,
        basis=basis,
        status=RecordStatusEnum.complete,
    ))
    # Defensive: even a "complete" record can have None properties if the
    # server is in an inconsistent state (e.g. mid-write). Skip those —
    # `result.return_result` dereferences properties and would crash.
    results = [r for r in results if r.properties is not None]
    if not results:
        logger.info(
            f"No complete hessian record at {method}/{basis} for molecule "
            f"{mol} ({mol_form})."
        )
        return None, True
    if len(results) > 1:
        logger.warning(
            f"Found {len(results)} complete hessian records for molecule "
            f"{mol} ({mol_form}) at {method}/{basis}; using the first."
        )
    result = results[0]
    logger.info(
        f"Molecule {mol} with molecular formula {mol_form} "
        "has a Hessian calculation."
    )

    import numpy as np
    hess_raw = result.return_result
    n_coords = 3 * len(result.molecule.symbols)
    hess = np.array(hess_raw).reshape(n_coords, n_coords)
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

    all_imag = [num for num in vib["omega"].data if abs(num.imag) > 1]
    small_imag = [f for f in all_imag if abs(f.imag) < imag_threshold]
    significant_imag = [f for f in all_imag if abs(f.imag) >= imag_threshold]

    if small_imag:
        logger.info(
            f"Molecule {mol}: {len(small_imag)} small imaginary "
            f"frequencies below {imag_threshold} cm⁻¹ ignored: "
            f"{[round(f.imag, 1) for f in small_imag]}"
        )

    if significant_imag:
        if on_imaginary == "raise":
            np.savetxt(f"{mol}_hess.dat", hess, fmt="%.18e")
            raise ValueError(
                f"There are {len(significant_imag)} significant imaginary "
                f"frequencies (>= {imag_threshold} cm⁻¹): {significant_imag}. "
                f"You need to reoptimize {mol}."
            )
        elif on_imaginary == "return":
            if len(significant_imag) > 1:
                logger.warning(
                    f"Molecule {mol}: {len(significant_imag)} significant "
                    f"imaginary frequencies: {significant_imag}. "
                    "Structure may be at a higher-order saddle point."
                )
            else:
                logger.info(
                    f"Molecule {mol}: 1 significant imaginary frequency: "
                    f"{significant_imag[0]}, proceed with caution."
                )
            return therm["ZPE_vib"].data, False
        else:
            raise ValueError(
                f"Invalid option for on_imaginary: {on_imaginary}"
            )

    return therm["ZPE_vib"].data, True


