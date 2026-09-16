"""SAPT workflow."""
import csv
import logging
import re
from pathlib import Path

from ..adapters import qcfractal_adapter as qcf
from ..core.sapt import build_sapt_molecule, extract_sapt_components
from ..models.base import safe_config_dump
from ..models.sapt import SaptConfig


ENTRY_CLUSTER_PATTERN = re.compile(
    r"^(.+)_(W[1-9][0-9]*_[0-9]+)_([0-9]+)$",
    re.IGNORECASE,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def sapt_dataset_name(config: SaptConfig) -> str:
    """Return the SinglepointDataset name for one SAPT workflow config."""
    parts = [
        "sapt",
        config.molecule,
        config.surface_model,
        config.optimization_spec,
        config.method,
        config.basis,
    ]
    return "_".join(_safe_name(part) for part in parts)


def sapt_spec_name(config: SaptConfig) -> str:
    """Return the SAPT singlepoint specification name."""
    return f"{config.method}_{config.basis}".lower()


def sapt_keywords(config: SaptConfig) -> dict[str, object]:
    """Return SAPT keywords, inferring UHF for open-shell fragments."""
    keywords = dict(config.keywords)
    has_reference = any(key.lower() == "reference" for key in keywords)
    open_shell = config.surface_multiplicity > 1 or config.molecule_multiplicity > 1
    if open_shell and not has_reference:
        keywords["reference"] = "uhf"
    return keywords


def explicit_requested_entries(requested) -> set[str] | None:
    """Return explicitly requested entries, or None when all are requested."""
    if requested == "all":
        return None
    if isinstance(requested, list):
        for entry in requested:
            entry_cluster(entry)
        return set(requested)
    if isinstance(requested, dict):
        entries = set()
        for cluster, chosen in requested.items():
            if not isinstance(chosen, list):
                raise ValueError(f"entries['{cluster}'] must be a list")
            for entry in chosen:
                if entry_cluster(entry) != cluster.upper():
                    raise ValueError(f"Entry '{entry}' does not belong to cluster {cluster.upper()}")
                entries.add(entry)
        return entries
    raise ValueError("entries must be 'all', a list, or a cluster-to-list mapping")


def entry_cluster(entry_name: str) -> str:
    """Return the water-cluster token encoded in a binding-site entry name."""
    match = ENTRY_CLUSTER_PATTERN.fullmatch(entry_name)
    if match is None:
        raise ValueError(f"Invalid SAPT entry name '{entry_name}'")
    return match.group(2).upper()


def requested_entries_for_cluster(requested, cluster: str) -> list[str] | None:
    """Return requested entries for one cluster, or None when all are requested."""
    cluster = cluster.upper()
    if requested == "all":
        return None
    if isinstance(requested, list):
        return [entry for entry in requested if entry_cluster(entry) == cluster]
    if isinstance(requested, dict):
        chosen = requested.get(cluster, requested.get(cluster.lower(), []))
        if not isinstance(chosen, list):
            raise ValueError(f"entries['{cluster}'] must be a list")
        for entry in chosen:
            if entry_cluster(entry) != cluster:
                raise ValueError(f"Entry '{entry}' does not belong to cluster {cluster}")
        return chosen
    raise ValueError("entries must be 'all', a list, or a cluster-to-list mapping")


def selected_entries_for_cluster(requested, cluster: str, available: list[str]) -> list[str]:
    """Select entries for one cluster from 'all', a list, or a mapping."""
    chosen = requested_entries_for_cluster(requested, cluster)
    if chosen is None:
        chosen = list(available)

    missing = sorted(set(chosen) - set(available))
    if missing:
        raise KeyError(f"Entries not found in cluster {cluster.upper()}: {', '.join(missing)}")
    return chosen


def _complete_record(record) -> bool:
    return record is not None and qcf.is_complete(record.status) and record.final_molecule is not None


def collect_fragmented_entries(config: SaptConfig, client, logger) -> list[tuple[str, object]]:
    """Collect fragmented SAPT molecules from completed optimization records."""
    surface_ds = qcf.get_collection(client, "OptimizationDataset", config.surface_model)
    exclusions = {cluster.upper() for cluster in config.exclude_clusters}
    fragmented_entries = []
    requested_entries = explicit_requested_entries(config.entries)
    seen_requested_entries: set[str] = set()

    for cluster_raw in surface_ds.entry_names:
        cluster = cluster_raw.upper()
        if cluster in exclusions:
            logger.info(f"{cluster}: excluded")
            continue

        preselected = requested_entries_for_cluster(config.entries, cluster)
        if preselected == []:
            logger.info(f"{cluster}: no SAPT entries requested")
            continue

        opt_ds_name = f"{config.molecule}_{cluster}"
        opt_ds = qcf.get_collection(client, "OptimizationDataset", opt_ds_name)
        if config.optimization_spec not in opt_ds.specification_names:
            raise KeyError(
                f"Specification '{config.optimization_spec}' not found in {opt_ds_name}; "
                f"available: {', '.join(opt_ds.specification_names)}"
            )

        selected = selected_entries_for_cluster(
            config.entries,
            cluster,
            list(opt_ds.entry_names),
        )
        complete = 0
        for entry_name in selected:
            if requested_entries is not None:
                seen_requested_entries.add(entry_name)
            record = opt_ds.get_record(entry_name, config.optimization_spec)
            if not _complete_record(record):
                logger.warning(f"Skipping {entry_name}: optimization is not COMPLETE")
                continue
            fragmented = build_sapt_molecule(
                record.final_molecule,
                cluster,
                surface_charge=config.surface_charge,
                surface_multiplicity=config.surface_multiplicity,
                molecule_charge=config.molecule_charge,
                molecule_multiplicity=config.molecule_multiplicity,
            )
            fragmented_entries.append((entry_name, fragmented))
            complete += 1
        logger.info(f"{cluster}: prepared {complete} SAPT entr{'y' if complete == 1 else 'ies'}")

    if requested_entries is not None:
        missing_requested = sorted(requested_entries - seen_requested_entries)
        if missing_requested:
            raise KeyError(
                "Requested SAPT entries were not found in the selected surface model: "
                + ", ".join(missing_requested)
            )

    return fragmented_entries


def _write_plan(path: Path, entries: list[tuple[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["entry", "n_atoms", "surface_atoms", "adsorbate_atoms"],
        )
        writer.writeheader()
        for entry_name, molecule in entries:
            surface_atoms = len(molecule.fragments[0])
            adsorbate_atoms = len(molecule.fragments[1])
            writer.writerow({
                "entry": entry_name,
                "n_atoms": len(molecule.symbols),
                "surface_atoms": surface_atoms,
                "adsorbate_atoms": adsorbate_atoms,
            })


def _write_results(path: Path, dataset, entries: list[tuple[str, object]], spec_name: str, config: SaptConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "entry",
        "status",
        "method",
        "basis",
        "electrostatics_kcal_mol",
        "exchange_kcal_mol",
        "induction_kcal_mol",
        "dispersion_kcal_mol",
        "total_sapt_kcal_mol",
        "message",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry_name, _ in entries:
            record = dataset.get_record(entry_name, spec_name)
            status = qcf.status_label(record.status) if record is not None else "MISSING"
            row = {
                "entry": entry_name,
                "status": status,
                "method": config.method,
                "basis": config.basis,
                "electrostatics_kcal_mol": "",
                "exchange_kcal_mol": "",
                "induction_kcal_mol": "",
                "dispersion_kcal_mol": "",
                "total_sapt_kcal_mol": "",
                "message": "",
            }
            if record is not None and qcf.is_complete(record.status):
                try:
                    row.update(extract_sapt_components(record, method=config.method))
                except KeyError as exc:
                    row["status"] = "PARSE_ERROR"
                    row["message"] = str(exc)
            writer.writerow(row)


def run(config: SaptConfig, client) -> None:
    logger = logging.getLogger("beep")
    res_folder = Path.cwd() / config.molecule / "sapt"
    res_folder.mkdir(parents=True, exist_ok=True)

    log_file = res_folder / f"sapt_{config.molecule}.log"
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    try:
        (res_folder / f"sapt_{config.molecule}.json").write_text(safe_config_dump(config))
        logger.info(f"Preparing SAPT workflow for {config.molecule}")
        entries = collect_fragmented_entries(config, client, logger)
        if not entries:
            logger.info("No completed optimization entries found. Nothing to process.")
            return

        plan_path = res_folder / "sapt_plan.csv"
        _write_plan(plan_path, entries)
        logger.info(f"Prepared {len(entries)} fragmented SAPT entr{'y' if len(entries) == 1 else 'ies'}")
        logger.info(f"Plan CSV: {plan_path}")

        if config.dry_run:
            logger.info("Dry run requested; no SAPT dataset was created and no jobs were submitted.")
            return

        dataset_name = sapt_dataset_name(config)
        spec_name = sapt_spec_name(config)
        dataset = qcf.get_or_create_singlepoint_dataset(client, dataset_name)
        qcf.add_singlepoint_entries(dataset, entries)
        qcf.add_energy_spec(
            dataset,
            spec_name=spec_name,
            method=config.method,
            basis=config.basis,
            program=config.program,
            keywords=sapt_keywords(config),
            description=f"{config.method}/{config.basis} SAPT energy decomposition",
        )
        meta = qcf.submit_singlepoints_in_dataset(
            dataset,
            spec_names=[spec_name],
            tag=config.sapt_tag,
            subset=[entry_name for entry_name, _ in entries],
        )
        logger.info(
            f"Submitted {getattr(meta, 'n_inserted', 0)} new SAPT records "
            f"({getattr(meta, 'n_existing', 0)} existing)."
        )

        if config.wait_for_completion:
            record_ids = []
            for entry_name, _ in entries:
                record = dataset.get_record(entry_name, spec_name)
                if record is not None:
                    record_ids.append(record.id)
            qcf.check_jobs_status(
                client,
                record_ids,
                logger,
                wait_interval=config.wait_interval,
            )

        _write_results(res_folder / "sapt_results.csv", dataset, entries, spec_name, config)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()
