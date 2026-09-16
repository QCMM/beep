"""MBE submission workflow — submit + monitor qcmanybody computations.

Ported from beep-mbe's ``submission.run``. Re-evaluates binding energies on
existing OptimizationDataset binding sites via n-body fragmentation on a
ManybodyDataset (plus a monomer SinglepointDataset for the isolated adsorbate).
The QCFractal client is supplied by the CLI; all server I/O goes through the
adapter.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from qcportal import PortalRequestError

from ..models.mbe import MbeConfig
from ..models.base import safe_config_dump
from ..core.logging_utils import padded_log, beep_banner
from ..core import mbe_fragmentation
from ..core.mbe_be_tools import (
    MbeSubmissionResult,
    manybody_dataset_name,
    monomer_dataset_name,
    submitted_entry_names,
)
from ..core.exceptions import MbeSubmissionError, MbeRecordMissingError
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient

bcheck = "✔"
gear = "⚙"

welcome_msg = beep_banner(
    "Many-Body Expansion Binding Energy Computation",
    quote="The whole is more than the sum of its parts.",
    quote_author="Aristotle",
    authors="Stefan Vogt-Geisse",
)


def _dataset_entries(dataset) -> List[str]:
    """List entry names of an OptimizationDataset (best effort)."""
    if getattr(dataset, "entry_names", None) is not None:
        return list(dataset.entry_names)
    return []


def _add_entry(mb_ds, entry_name, molecule, update_existing, existing_entries) -> None:
    """Add or update a ManybodyDataset entry for a fragmented cluster."""
    logger = logging.getLogger("beep")
    if entry_name in existing_entries and not update_existing:
        logger.info(f"Entry '{entry_name}' exists; skipping (update_existing_entries=false).")
        return
    try:
        mb_ds.add_entry(name=entry_name, initial_molecule=molecule, overwrite=update_existing)
    except TypeError:
        mb_ds.add_entry(name=entry_name, initial_molecule=molecule)
    logger.info(f"Entry added/updated: {entry_name}")


def _fetch_record_summary(mb_ds, entry, spec, show_children) -> str:
    """Fetch and summarize record status for a ManybodyDataset entry."""
    logger = logging.getLogger("beep")
    try:
        mb_ds.fetch_records(
            entry_names=[entry],
            specification_names=[spec],
            include=["properties", "initial_molecule", "clusters"],
        )
    except Exception as exc:
        logger.debug(f"fetch_records failed for {entry}/{spec}: {exc}")
    rec = mb_ds.get_record(entry_name=entry, specification_name=spec)
    if rec is None:
        logger.info(f"No record found yet for entry '{entry}' and spec '{spec}'.")
        return "missing"

    logger.info(f"Record status for {entry}: {rec.status}")
    props = rec.properties or {}
    candidates = [
        "manybody_total_energy_cp",
        "manybody_total_energy_vmfc",
        "manybody_total_energy_nocp",
        "supersystem_energy_cp",
        "supersystem_energy_vmfc",
    ]
    found_any = False
    for key in candidates:
        if key in props:
            found_any = True
            logger.info(f"{key} = {props[key]}")
    if not found_any:
        logger.info(f"No expected total-energy properties found yet. Available keys: {sorted(props)}")
    return str(rec.status)


def submit_mbe(config: MbeConfig, client: FractalClient) -> MbeSubmissionResult:
    """Submit and (optionally) monitor MBE computations. Testable core of run()."""
    logger = logging.getLogger("beep")

    opt_ds = qcf.get_collection(client, "OptimizationDataset", config.opt_dataset)
    entry_names = list(config.entries) if config.entries else _dataset_entries(opt_ds)
    if not entry_names:
        raise MbeSubmissionError("No optimization entries available for submission.")
    if config.small_molecule:
        entry_names = [n for n in entry_names if n != config.small_molecule]

    submission_entry_names = submitted_entry_names(
        config.surface_model, config.small_molecule, entry_names
    )
    mb_ds_name = manybody_dataset_name(config.dataset, config.opt_level_of_theory)

    padded_log(logger, "Starting Many-Body Expansion submission", padding_char=gear)
    levels_str = ", ".join(
        f"{lvl.index}-body: {lvl.method}/{lvl.basis}" for lvl in config.levels
    )
    mbe_parameters = (
        f"Many-Body Expansion Computation Parameters:\n"
        f"- Small Molecule: {config.small_molecule}\n"
        f"- Surface Model: {config.surface_model}\n"
        f"- Optimization Dataset: {config.opt_dataset}\n"
        f"- Optimization Level of Theory: {config.opt_level_of_theory}\n"
        f"- ManybodyDataset: {mb_ds_name}\n"
        f"- Specifications: {' '.join(config.spec)}\n"
        f"- MBE Levels: {levels_str}\n"
        f"- BSSE Correction: {' '.join(config.bsse)}\n"
        f"- Environment Fragment Size: {config.env_unit_len} atoms\n"
        f"- Cluster Entries: {len(entry_names)}\n"
        f"- Program: {config.program}\n"
        f"- Compute Tag: {config.tag}\n"
    )
    logger.info(mbe_parameters)

    padded_log(logger, "Fetching and fragmenting reference geometries", padding_char=gear)
    small_ds = qcf.get_collection(client, "OptimizationDataset", config.small_molecule_collection)
    surface_ds = qcf.get_collection(client, "OptimizationDataset", config.surface_model_collection)

    try:
        small_mol_raw = qcf.fetch_final_molecule(small_ds, config.small_molecule, config.opt_level_of_theory)
        surface_mol_raw = qcf.fetch_final_molecule(surface_ds, config.surface_model, config.opt_level_of_theory)
    except KeyError as exc:
        raise MbeRecordMissingError(
            f"Missing optimized reference molecule at {config.opt_level_of_theory}: {exc}"
        ) from exc

    small_mol = mbe_fragmentation.fragment_small_molecule(small_mol_raw)
    surface_mol = mbe_fragmentation.fragment_surface_model(surface_mol_raw, config.env_unit_len)

    logger.info(f"Small molecule atoms: {len(small_mol.symbols)}")
    logger.info(f"Surface model atoms: {len(surface_mol.symbols)}")
    logger.info(f"Surface fragments: {len(surface_mol.fragments)}")

    padded_log(logger, "Registering ManybodyDataset specifications and entries", padding_char=gear)
    mb_ds = qcf.get_or_create_manybody_dataset(client, mb_ds_name)

    specs = config.spec
    levels = qcf.mbe_levels_to_qc_specifications(config.levels, config.program)
    mb_spec = qcf.build_manybody_specification(levels, config.bsse)
    for spec in specs:
        mb_ds.add_specification(spec, mb_spec)
        logger.info(f"Manybody specification registered: {spec}")
    logger.info(f"BSSE correction: {config.bsse}")

    monomer_spec = levels.get(1)
    if monomer_spec is None:
        raise MbeSubmissionError("No 1-body level found in configuration levels.")

    monomer_spec_names = [f"monomer_{spec}" for spec in specs]
    sp_ds_name = monomer_dataset_name(config.small_molecule_collection, config.opt_level_of_theory)
    sp_ds = qcf.get_or_create_singlepoint_dataset(client, sp_ds_name)
    logger.info(f"Using SinglepointDataset for monomers: {sp_ds_name}")

    sp_specs = getattr(sp_ds, "specifications", {}) or {}
    for monomer_spec_name in monomer_spec_names:
        if monomer_spec_name not in sp_specs:
            sp_ds.add_specification(monomer_spec_name, monomer_spec)
            logger.info(f"Singlepoint specification registered: {monomer_spec_name}")

    sp_existing = list(getattr(sp_ds, "entry_names", None) or [])
    if config.small_molecule not in sp_existing:
        sp_ds.add_entry(name=config.small_molecule, molecule=small_mol_raw)
        logger.info(f"Singlepoint entry added: {config.small_molecule}")
    else:
        logger.info(f"Singlepoint entry exists; skipping add: {config.small_molecule}")

    existing_entries = list(getattr(mb_ds, "entry_names", None) or [])
    _add_entry(mb_ds, config.surface_model, surface_mol, config.update_existing_entries, existing_entries)

    for entry_name in entry_names:
        try:
            cluster_mol_raw = qcf.fetch_final_molecule(opt_ds, entry_name, config.opt_level_of_theory)
        except KeyError as exc:
            raise MbeRecordMissingError(
                f"Missing optimized cluster '{entry_name}' at {config.opt_level_of_theory}: {exc}"
            ) from exc
        cluster_mol = mbe_fragmentation.fragment_cluster(
            cluster_mol_raw, config.env_unit_len, len(small_mol.symbols)
        )
        logger.info(
            f"Cluster '{entry_name}' fragments: {len(cluster_mol.fragments)} "
            f"(env_unit_len={config.env_unit_len})"
        )
        _add_entry(mb_ds, entry_name, cluster_mol, config.update_existing_entries, existing_entries)

    submitted_metadata: Optional[object] = None
    if not config.fetch_only:
        padded_log(logger, "Submitting Many-Body Expansion computations", padding_char=gear)
        logger.info(f"Submitting many-body jobs with compute tag: {config.tag}")
        try:
            submitted_metadata = mb_ds.submit(
                entry_names=submission_entry_names,
                specification_names=specs,
                compute_tag=config.tag,
                find_existing=True,
            )
        except TypeError:
            submitted_metadata = mb_ds.submit(
                entry_names=submission_entry_names,
                specification_names=specs,
                compute_tag=config.tag,
            )
        except PortalRequestError as exc:
            logger.error(f"Submission failed: {exc}")
            raise MbeSubmissionError("Submission failed; see logs for details.") from exc
        logger.info(f"Submission metadata: {submitted_metadata}")

    monomer_submission_entries: List[str] = []
    monomer_submission_specs: List[str] = []
    if not config.fetch_only:
        for monomer_spec_name in monomer_spec_names:
            monomer_record = sp_ds.get_record(
                entry_name=config.small_molecule, specification_name=monomer_spec_name
            )
            # Use the adapter predicate: str(RecordStatusEnum.complete) is
            # "RecordStatusEnum.complete", so a string comparison never matches
            # (latent bug inherited from beep-mbe).
            if monomer_record is not None and qcf.is_complete(monomer_record.status):
                logger.info("Monomer singlepoint already COMPLETE; skipping submit.")
                continue
            monomer_submission_specs.append(monomer_spec_name)

        if monomer_submission_specs:
            logger.info(f"Submitting monomer singlepoint with compute tag: {config.tag}")
            try:
                sp_ds.submit(
                    entry_names=[config.small_molecule],
                    specification_names=monomer_submission_specs,
                    compute_tag=config.tag,
                    find_existing=True,
                )
            except TypeError:
                sp_ds.submit(
                    entry_names=[config.small_molecule],
                    specification_names=monomer_submission_specs,
                    compute_tag=config.tag,
                )
            monomer_submission_entries.append(config.small_molecule)

    statuses: Dict[Tuple[str, str], str] = {}
    for entry_name in submission_entry_names:
        for spec in specs:
            statuses[(entry_name, spec)] = _fetch_record_summary(
                mb_ds, entry_name, spec, config.show_children
            )

    monitor_result: Optional[Dict[str, object]] = None
    if config.monitor.enabled and not config.fetch_only:
        padded_log(logger, "Monitoring Many-Body Expansion computations", padding_char=gear)
        monitor_result = {}
        timed_out_specs: List[str] = []
        errored_entries: List[str] = []
        for spec in specs:
            result = qcf.wait_for_manybody_completion(
                client=client,
                dataset_name=mb_ds_name,
                spec_name=spec,
                entry_names=submission_entry_names,
                poll_interval_s=config.monitor.poll_interval,
                max_wait_s=config.monitor.max_wait,
            )
            monitor_result[spec] = result
            if result.timed_out:
                timed_out_specs.append(spec)
            errored_entries.extend(result.errored_entries)

        monomer_timed_out = False
        monomer_errored: List[str] = []
        if monomer_submission_entries:
            monomer_statuses, monomer_timed_out = qcf.wait_for_dataset_records(
                sp_ds,
                entry_names=monomer_submission_entries,
                specification_names=monomer_submission_specs,
                poll_interval=config.monitor.poll_interval,
                max_wait=config.monitor.max_wait,
            )
            monomer_errored = [
                entry for (entry, _spec), status in monomer_statuses.items() if status == "ERROR"
            ]

        if timed_out_specs:
            raise MbeSubmissionError(
                f"Monitoring timed out for specs: {', '.join(timed_out_specs)}."
            )
        if monomer_timed_out:
            raise MbeSubmissionError("Monomer monitoring timed out before completion.")
        if errored_entries:
            raise MbeSubmissionError(
                f"Monitoring finished with errored entries: {', '.join(errored_entries)}"
            )
        if monomer_errored:
            raise MbeSubmissionError(
                f"Monomer monitoring finished with errored entries: {', '.join(monomer_errored)}"
            )

    return MbeSubmissionResult(
        dataset_name=mb_ds_name,
        specification_names=list(specs),
        entry_names=submission_entry_names,
        submitted_metadata=submitted_metadata,
        fetched_statuses=statuses,
        monitor_result=monitor_result,
    )


def run(config: MbeConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    res_folder = Path.cwd() / config.small_molecule
    res_folder.mkdir(parents=True, exist_ok=True)

    log_file = res_folder / f"mbe_{config.small_molecule}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    config_path = res_folder / f"mbe_{config.small_molecule}.json"
    config_path.write_text(safe_config_dump(config))

    logger.info(welcome_msg)
    try:
        result = submit_mbe(config, client)
        logger.info(
            f"\nMBE submission complete for dataset '{result.dataset_name}': "
            f"{len(result.entry_names)} entries, specs {result.specification_names}. {bcheck}"
        )
        logger.info(
            "\nThank you for using the many-body expansion binding energy compute suite!"
        )
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()
