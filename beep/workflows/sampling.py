"""Sampling workflow — refactored from workflows/launch_sampling.py."""
import json
import logging
from pathlib import Path

from ..models.sampling import SamplingConfig
from ..core.logging_utils import padded_log, beep_banner
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient

bcheck = "\u2714"
gear = "\u2699"

welcome_msg = beep_banner(
    "Set-of-clusters Sampling",
    quote="Adopt the pace of nature: her secret is patience.",
    quote_author="Ralph Waldo Emerson",
    tagline="Seek Locate Map",
    authors="Stefan Vogt-Geisse and Giulia M. Bovolenta",
)


def config_summary_msg(config):
    """Format a clean summary of the sampling configuration."""
    separator = "-" * 88
    s_lot = config.sampling_level_of_theory
    r_lot = config.refinement_level_of_theory
    s_basis = s_lot.basis or "N/A"
    lines = [
        "",
        separator,
        f"  Molecule:             {config.molecule}",
        f"  Surface model:        {config.surface_model_collection}",
        f"  Small molecule coll:  {config.small_molecule_collection}",
        f"  Sampling LOT:         {s_lot.method}/{s_basis} ({s_lot.program})",
        f"  Refinement LOT:       {r_lot.method}/{r_lot.basis} ({r_lot.program})",
        f"  Sampling shell:       {config.sampling_shell} Angstrom",
        f"  Sampling condition:   {config.sampling_condition}",
        f"  RMSD cutoff:          {config.rmsd_value}",
        f"  RMSD symmetry:        {config.rmsd_symmetry}",
        f"  Target binding sites: {config.total_binding_sites}",
        separator,
        "",
    ]
    return "\n".join(lines)


def process_refinement(client, ropt_lot_name, rmethod, rbasis, program,
                       qc_keyword, ds_opt, logger, rtag="refinement"):
    spec = {
        "name": ropt_lot_name,
        "description": f"Geometric + {rmethod}/{rbasis}/{program}",
        "optimization_spec": {"program": "geometric", "keywords": None},
        "qc_spec": {
            "driver": "gradient",
            "method": rmethod,
            "basis": rbasis,
            "keywords": qc_keyword,
            "program": program,
        },
    }

    qcf.add_opt_specification(ds_opt, spec, overwrite=True)
    c = qcf.submit_optimizations(ds_opt, ropt_lot_name, tag=rtag)

    logger.info(
        f"\nRefinement optimization initiated with specification '{ropt_lot_name}' \n"
        f"using {rmethod}/{rbasis} in {program}. \n"
        f"Description: {spec['description']}. \n"
        f"Tag applied: '{rtag}'\n"
        f"Optimizations submitted: {c}. {bcheck} \n"
    )


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
    target_mol,
    cluster,
    debug_path,
    client,
    sampling_shell: float,
    sampling_condition: str,
    logger,
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

    # Build specification dict for the adapter
    spec = {
        "name": opt_lot,
        "description": "Geometric Optimization",
        "optimization_spec": {"program": "geometric", "keywords": {"maxiter": 125}},
        "qc_spec": {
            "driver": "gradient",
            "method": method,
            "basis": basis,
            "keywords": kw_id if isinstance(kw_id, dict) else {},
            "program": program,
        },
    }
    qcf.add_opt_specification(sampling_opt_dset, spec, overwrite=False)
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

        pid_list = qcf.get_job_ids(sampling_opt_dset, shell_old_entries, opt_lot)

        if pid_list:
            logger.debug(
                f"Procedure IDs of the optimization are: {pid_list}"
            )

        qcf.wait_for_completion(client, pid_list, FREQUENCY, logger)

        added_names = []
        if not shell_new_entries:
            logger.info(
                "All entries for this shell already exist."
            )
        else:
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
                name = shell_new_entries[i]
                try:
                    qcf.add_opt_entry(sampling_opt_dset, name, m)
                    added_names.append(name)
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

        # Always submit at the current sampling LOT for the union of newly-
        # added entries plus entries already in the dataset (from a prior
        # run, possibly at a different LOT). QCFractal dedups server-side:
        # entries already computed at this spec become n_existing no-ops;
        # entries that have never been computed at this spec get fresh opts.
        present_entries = shell_old_entries + added_names
        if present_entries:
            comp_rec = qcf.submit_optimizations(
                sampling_opt_dset, opt_lot, tag=tag, subset=present_entries,
            )
            logger.info(
                f"{comp_rec.n_inserted} new, {comp_rec.n_existing} existing "
                "sampling optimization procedures."
            )

        pid_list = qcf.get_job_ids(sampling_opt_dset, present_entries, opt_lot)
        if pid_list:
            logger.info(
                "Procedure IDs of the optimization are: {}".format(
                    " ".join(str(p) for p in pid_list)
                )
            )

        qcf.wait_for_completion(client, pid_list, FREQUENCY, logger)

        opt_molecules_new = qcf.fetch_opt_molecules(
            sampling_opt_dset, present_entries, opt_lot, status="COMPLETE"
        )
        opt_mol_num = len(opt_molecules_new)
        logger.debug(
            f"{opt_mol_num} COMPLETED molecules in "
            f"{sampling_opt_dset.name} for this round, "
            f"{len(present_entries) - opt_mol_num} molecules ended in ERROR."
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
                qcf.add_opt_entry(
                    refinement_opt_dset, entry_name, mol_obj,
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


def run(config: SamplingConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    smol_name = config.molecule

    # Create output folder: <molecule>/sampling/
    res_folder = Path.cwd() / smol_name / "sampling"
    res_folder.mkdir(parents=True, exist_ok=True)

    # File logging inside the output folder
    log_file = res_folder / f"beep_sampling_{smol_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Save a copy of the input config
    config_path = res_folder / f"sampling_{smol_name}.json"
    config_path.write_text(json.dumps(config.dict(), indent=4, default=str))

    logger.info(welcome_msg)

    method = config.sampling_level_of_theory.method
    basis = config.sampling_level_of_theory.basis
    program = config.sampling_level_of_theory.program
    rmethod = config.refinement_level_of_theory.method
    rbasis = config.refinement_level_of_theory.basis
    rprogram = config.refinement_level_of_theory.program

    qc_keyword = config.keyword_id

    if basis:
        opt_lot = (method + "_" + basis).lower()
    else:
        opt_lot = method.lower()
    ropt_lot = (rmethod + "_" + rbasis).lower()

    # --- Configuration summary ---
    logger.info(config_summary_msg(config))

    args_dict = {
        "method": method,
        "basis": basis,
        "program": program,
        "tag": config.sampling_tag,
        "kw_id": None,  # keyword_id is used for refinement spec, not sampling
        "rmsd_symm": config.rmsd_symmetry,
        "store_initial": config.store_initial_structures,
        "rmsd_val": config.rmsd_value,
        "sampling_shell": config.sampling_shell,
        "sampling_condition": config.sampling_condition,
        "opt_lot": opt_lot,
        "logger": logger,
    }

    # --- Validate collections ---
    logger.info("Validating collections and optimized geometries...")
    qcf.check_collection_existence(
        client, config.surface_model_collection, config.small_molecule_collection
    )
    ds_sm = qcf.get_collection(client, "OptimizationDataset", config.small_molecule_collection)
    ds_wc = qcf.get_collection(client, "OptimizationDataset", config.surface_model_collection)

    all_cluster_names = list(ds_wc.entry_names)
    if config.sampling_clusters:
        cluster_names = [c for c in all_cluster_names if c in config.sampling_clusters]
        logger.info(
            f"  Surface model clusters: {len(all_cluster_names)} available, "
            f"{len(cluster_names)} selected  ({', '.join(cluster_names)})"
        )
    else:
        cluster_names = all_cluster_names
        logger.info(f"  Surface model clusters: {len(cluster_names)}  ({', '.join(cluster_names)})")

    # Check if all the molecules are optimized at the requested level of theory.
    # Atoms (single-atom species) live in a SinglepointDataset, not an
    # OptimizationDataset, so we fetch them from the atoms_collection.
    try:
        init_mol = qcf.fetch_initial_molecule(ds_sm, smol_name, opt_lot)
        is_atom = len(init_mol.symbols) == 1
    except KeyError:
        # Molecule not in the OptimizationDataset — try the atoms collection
        init_mol = qcf.fetch_atom_molecule(client, config.atoms_collection, smol_name)
        is_atom = True

    if is_atom:
        qcf.check_optimized_molecule(ds_wc, opt_lot, cluster_names)
        args_dict["target_mol"] = init_mol
    else:
        qcf.check_optimized_molecule(ds_sm, opt_lot, [smol_name])
        qcf.check_optimized_molecule(ds_wc, opt_lot, cluster_names)
        args_dict["target_mol"] = qcf.fetch_final_molecule(ds_sm, smol_name, opt_lot)

    logger.info(f"  All geometries validated. {bcheck}\n")

    args_dict["client"] = client

    # --- Sampling loop ---
    count = 0
    cluster_results = []
    refinement_dsets = []

    for c, w in enumerate(cluster_names):
        args_dict["cluster"] = qcf.fetch_final_molecule(ds_wc, w, opt_lot)

        ref_opt_dset_name = smol_name + "_" + w
        smplg_opt_dset_name = "pre_" + ref_opt_dset_name

        ds_smplg = qcf.get_or_create_opt_dataset(client, smplg_opt_dset_name)
        ds_ref = qcf.get_or_create_opt_dataset(client, ref_opt_dset_name)
        args_dict["sampling_opt_dset"] = ds_smplg
        args_dict["refinement_opt_dset"] = ds_ref

        logger.info(f"\n{'=' * 80}")
        logger.info(f"  Cluster {c+1}/{len(cluster_names)}: {w}")
        logger.info(f"  Sampling dataset:    {smplg_opt_dset_name}")
        logger.info(f"  Refinement dataset:  {ref_opt_dset_name}")
        logger.info(f"{'=' * 80}\n")

        debug_path = res_folder / "site_finder" / (str(smol_name) + "_w") / w
        if not debug_path.exists() and config.store_initial_structures:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
        args_dict["debug_path"] = debug_path

        run_sampling(**args_dict)

        process_refinement(
            client, ropt_lot, rmethod, rbasis, rprogram,
            qc_keyword, ds_ref, logger, config.refinement_tag,
        )

        ds_ref = qcf.get_or_create_opt_dataset(client, ref_opt_dset_name)
        n_sites = len(ds_ref.entry_names)
        count += n_sites
        cluster_results.append((w, n_sites))
        refinement_dsets.append((w, ds_ref))

        logger.info(f"\n  {bcheck} Cluster {w}: {n_sites} binding sites  |  Running total: {count}")

        if count > config.total_binding_sites:
            logger.info(f"\n  Target of {config.total_binding_sites} binding sites reached. Stopping early.")
            break

    # --- Wait for refinement optimizations to finish ---
    REFINEMENT_POLL_FREQUENCY = 120
    REFINEMENT_MAX_WAIT = 7 * 24 * 3600  # one week
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  Waiting for refinement optimizations to complete ({ropt_lot})")
    logger.info(f"{'=' * 80}\n")
    for w, ds_ref in refinement_dsets:
        entries = list(ds_ref.entry_names)
        pids = qcf.get_job_ids(ds_ref, entries, ropt_lot)
        if not pids:
            logger.info(f"  {w}: no refinement records to wait on.")
            continue
        logger.info(f"  {w}: waiting on {len(pids)} refinement opt(s).")
        qcf.wait_for_completion(
            client, pids, REFINEMENT_POLL_FREQUENCY, logger,
            max_wait=REFINEMENT_MAX_WAIT,
        )
        logger.info(f"  {w}: refinement complete. {bcheck}")

    # --- Final summary ---
    logger.info(f"\n\n{'=' * 80}")
    logger.info(f"  SAMPLING SUMMARY FOR {smol_name}")
    logger.info(f"{'=' * 80}")
    logger.info(f"  {'Cluster':<15} {'Binding sites':>15}")
    logger.info(f"  {'-'*15} {'-'*15}")
    for w, n in cluster_results:
        logger.info(f"  {w:<15} {n:>15}")
    logger.info(f"  {'-'*15} {'-'*15}")
    logger.info(f"  {'TOTAL':<15} {count:>15}")
    logger.info(f"{'=' * 80}\n")

    logger.removeHandler(file_handler)
    file_handler.close()
