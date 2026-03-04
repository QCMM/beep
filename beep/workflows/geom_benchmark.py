"""Geometry benchmark workflow — refactored from workflows/launch_geom_benchmark.py."""
import json
import time
import logging
import warnings
from typing import Dict, List, Tuple, Union
from collections import Counter

import numpy as np
import pandas as pd
import qcportal as ptl
from pathlib import Path
from qcportal.client import FractalClient
from qcportal.collections import OptimizationDataset, ReactionDataset
from qcelemental.models.molecule import Molecule

from ..models.geom_benchmark import GeomBenchmarkConfig
from ..core.logging_utils import (
    padded_log, log_dataframe_averages, log_progress, dict_to_log, beep_banner,
)
from ..core.dft_functionals import (
    geom_hmgga_dz, geom_hmgga_tz, geom_gga_dz, geom_sqm_mb,
)
from ..core.plotting_utils import rmsd_histograms
from ..core.benchmark_utils import create_benchmark_dataset_dict, compute_rmsd
from ..adapters import qcfractal_adapter as qcf

bcheck = "\u2714"
mia0911 = "\u2606"
gear = "\u2699"

welcome_msg = beep_banner(
    "Geometry Benchmark",
    tagline="Shine, Loom, Manifest.",
    authors="Stefan Vogt-Geisse",
)


def create_and_add_specification(client, odset, method, basis, program,
                                  qc_keyword, geom_keywords=None):
    logger = logging.getLogger("beep")
    spec_name = f"{method}_{basis}"
    if qc_keyword:
        kw_name = client.query_keywords()[qc_keyword].values.values()
        logger.debug(f"Using the following keyword for the specification {kw_name} to {odset.name}")
        if ("uks" in kw_name or "uhf" in kw_name) and program == "psi4":
            spec_name = "U" + spec_name

    spec = {
        "name": spec_name,
        "description": f"Geometric {program}/{method}/{basis}",
        "optimization_spec": {"program": "geometric", "keywords": geom_keywords},
        "qc_spec": {
            "driver": "gradient",
            "method": method,
            "basis": basis,
            "keywords": qc_keyword,
            "program": program,
        },
    }
    odset.add_specification(**spec, overwrite=True)
    odset.save()
    logger.debug(f"Create and added the specification {spec_name} to {odset.name}")
    return spec_name


def optimize_reference_molecule(odset, struct_name, geom_ref_opt_lot, mol_mult, opt_tag):
    if mol_mult in (1, 2):
        return qcf.submit_optimizations(odset, geom_ref_opt_lot, tag=opt_tag, subset={struct_name})
    else:
        raise RuntimeError(
            "Invalid value for molecular multiplicity. It has to be 1 (Singlet) or 2 (Doublet)"
        )


def optimize_dft_molecule(client, odset, struct_name, method, basis, program,
                           dft_keyword, opt_tag):
    spec_name = create_and_add_specification(client, odset, method, basis, program, dft_keyword)
    return qcf.submit_optimizations(odset, spec_name, tag=opt_tag, subset={struct_name})


def wait_for_completion(client, odset_dict, opt_lot, program, qc_keyword=None,
                         wait_interval=600, check_errors=False):
    logger = logging.getLogger("beep")
    if isinstance(opt_lot, str):
        opt_lot = [opt_lot]

    logger.info("\nChecking if the computations have finished")
    logger.info("\n")
    while True:
        statuses = []
        for lot in opt_lot:
            try:
                if ("uks" in client.query_keywords()[qc_keyword].values.values() or "uhf" in client.query_keywords()[qc_keyword].values.values()) and program == "psi4":
                    lot = "U" + lot
            except TypeError:
                pass
            for struct_name, odset in odset_dict.items():
                status = odset.get_record(struct_name, specification=lot).status
                statuses.append(status)

                if status == "ERROR" and check_errors:
                    raise RuntimeError(
                        f"Error encountered in computation for {struct_name} with spec '{lot}'"
                    )

        status_counts = Counter(statuses)

        if status_counts["INCOMPLETE"] == 0:
            logger.info(
                f"All entries have been processed. (Complete: {status_counts['COMPLETE']}, "
                f"ERROR: {status_counts['ERROR']}) {bcheck}"
            )
            return status_counts["COMPLETE"]

        logger.info(
            f"Waiting for {wait_interval} seconds before rechecking statuses... "
            f"(Incomplete: {status_counts['INCOMPLETE']})"
        )
        time.sleep(wait_interval)


def compare_rmsd(dft_lot, odset_dict, ref_geom_fmols):
    logger = logging.getLogger("beep")
    logger.propagate = False
    rmsd_df = pd.DataFrame(index=odset_dict.keys(), columns=dft_lot)
    final_opt_lot = {}
    total_operations = len(dft_lot)

    errored_specs = []
    for i, opt_lot in enumerate(dft_lot):
        rmsd_tot_dict = {}
        err = None
        for struct_name, odset in odset_dict.items():
            record = odset.get_record(struct_name, specification=opt_lot)
            err = record.get_error()
            if err:
                logger.warning(
                    f"WARNING: Calculation for {struct_name} at the {opt_lot} level of theory "
                    f"finished with error (record id: {record.id}). "
                    f"This level of theory will be excluded from the benchmark."
                )
                errored_specs.append((opt_lot, struct_name, record.id))
                break
            fmol = record.get_final_molecule()
            rmsd = compute_rmsd(ref_geom_fmols[struct_name], fmol, rmsd_symm=True)
            rmsd_tot_dict[struct_name] = rmsd
            rmsd_df.at[struct_name, opt_lot] = rmsd

        if err:
            rmsd_df[opt_lot] = np.nan
            continue
        rmsd_tot = list(rmsd_tot_dict.values())
        final_opt_lot[opt_lot] = np.mean(rmsd_tot)
        log_progress(logger, i + 1, total_operations)

    if errored_specs:
        logger.warning(f"\nSummary of errored optimizations ({len(errored_specs)} total):")
        for spec, struct, rec_id in errored_specs:
            logger.warning(f"  {spec} / {struct}  (record id: {rec_id})")
        logger.warning("")

    rmsd_df = rmsd_df.dropna(axis=1, how="all")
    lowest_values = sorted(final_opt_lot.values())[:1]
    best_geom_lot = {k: v for k, v in final_opt_lot.items() if v in lowest_values}
    return best_geom_lot, final_opt_lot, rmsd_df


def compare_all_rmsd(functional_groups, odset_dict, ref_geom_fmols):
    logger = logging.getLogger("beep")
    best_opt_lot = {}
    combined_rmsd_df = pd.DataFrame()

    for func_group, functionals in functional_groups.items():
        logger.info(f"\nProcessing RMSD for {func_group} type methods:")
        group_best_opt_lot, final_opt_lot, rmsd_df = compare_rmsd(
            functionals, odset_dict, ref_geom_fmols
        )
        rmsd_df.columns = [f"{func_group}_{col}" for col in rmsd_df.columns]
        combined_rmsd_df = pd.concat([combined_rmsd_df, rmsd_df], axis=1)
        best_opt_lot[func_group] = group_best_opt_lot

    return best_opt_lot, combined_rmsd_df


def run(config: GeomBenchmarkConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    smol_name = config.molecule

    # Create output folder: <molecule>/geom_benchmark/
    res_folder = Path.cwd() / smol_name / "geom_benchmark"
    res_folder.mkdir(parents=True, exist_ok=True)

    # File logging inside the output folder
    log_file = res_folder / f"beep_geom_benchmark_{smol_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Save a copy of the input config
    config_path = res_folder / f"geom_benchmark_{smol_name}.json"
    config_path.write_text(json.dumps(config.dict(), indent=4, default=str))

    logger.info(welcome_msg)

    hl_tag = config.tag_reference_geometry
    dft_tag = config.tag_dft_geometry
    gr_method, gr_basis, gr_program = config.reference_geometry_level_of_theory
    geom_ref_opt_lot = gr_method + "_" + gr_basis

    bchmk_structs = config.benchmark_structures
    surf_dset_name = config.surface_model_collection
    smol_dset_name = config.small_molecule_collection

    padded_log(logger, "Starting BEEP geometry benchmark procedure", padding_char=gear)
    logger.info(f"Molecule: {smol_name}")
    logger.info(f"Surface Model: {smol_dset_name}")
    logger.info(f"Benchmark Structures: {bchmk_structs}")

    odset_dict = {}
    bchmk_dset_names = create_benchmark_dataset_dict(bchmk_structs)

    qcf.check_collection_existence(client, *bchmk_dset_names.values())
    qcf.check_collection_existence(client, smol_dset_name)
    qcf.check_collection_existence(client, surf_dset_name)

    smol_dset = qcf.get_collection(client, "OptimizationDataset", smol_dset_name)
    mol_mult = qcf.get_molecular_multiplicity(client, smol_dset, smol_name)
    logger.info(f"\n The molecular multiplicity of {smol_name} is {mol_mult}")

    odset_dict = {smol_name: smol_dset}
    for bchmk_struct_name, odset_name in bchmk_dset_names.items():
        odset_dict[bchmk_struct_name] = qcf.get_collection(client, "OptimizationDataset", odset_name)
        surf_mod = bchmk_struct_name.split("_")[1].upper()
        odset_dict[surf_mod] = qcf.get_collection(client, "OptimizationDataset", surf_dset_name)

    padded_log(logger, "Start of the geometry refrence processing")
    logger.info(f"Method: {gr_method}")
    logger.info(f"Basis: {gr_basis}")
    logger.info(f"Program: {gr_program}\n")
    for odset in odset_dict.values():
        create_and_add_specification(
            client, odset, method=gr_method, basis=gr_basis,
            program=gr_program, qc_keyword=None, geom_keywords=None,
        )

    ct = 0
    for struct_name, odset in odset_dict.items():
        ct += optimize_reference_molecule(odset, struct_name, geom_ref_opt_lot, mol_mult, hl_tag)

    logger.info(
        f"\nSend a total of {ct} structures to compute at the "
        f"{geom_ref_opt_lot} level of theory to the tag {hl_tag}\n"
    )

    padded_log(logger, "Start of the DFT geometry computations")

    dft_program = config.dft_optimization_program
    dft_keyword = config.dft_optimization_keyword

    dft_geom_functionals = {
        "geom_hmgga_dz": geom_hmgga_dz(),
        "geom_hmgga_tz": geom_hmgga_tz(),
        "geom_gga_dz": geom_gga_dz(),
        "geom_sqm_mb": geom_sqm_mb(),
    }

    all_dft_functionals = [
        functional
        for functionals in dft_geom_functionals.values()
        for functional in functionals
    ]

    logger.info(f"Program: {dft_program}")
    logger.info(f"DFT and SQM geometry methods:")
    dict_to_log(logger, dft_geom_functionals)

    ct = 0
    c = 0
    padded_log(logger, "Start sending DFT optimizations")
    for struct_name, odset in odset_dict.items():
        logger.info(f"\nSending geometry optimizations for {struct_name}")
        cs = 0
        for functionals in dft_geom_functionals.values():
            for functional in functionals:
                method, basis = functional.split("_")
                cs += optimize_dft_molecule(
                    client, odset, struct_name, method, basis,
                    dft_program, dft_keyword, dft_tag,
                )
                ct += cs
                c += 1
        logger.info(f"Send {cs} geometry optimizations for structure {struct_name}")

    logger.info(f"\nSend {c}/{ct} to the tag {dft_tag}")

    wait_for_completion(
        client, odset_dict, all_dft_functionals, dft_program, dft_keyword,
        wait_interval=200, check_errors=False,
    )

    ref_geom_fmols = {}
    for struct_name, odset in odset_dict.items():
        record = odset.get_record(struct_name, specification=geom_ref_opt_lot)
        if config.use_initial_reference_geometry:
            ref_geom_fmols[struct_name] = record.get_initial_molecule()
        else:
            ref_geom_fmols[struct_name] = record.get_final_molecule()

    padded_log(
        logger,
        "Start of RMSD comparison between DFT and {} geometries",
        geom_ref_opt_lot,
    )

    try:
        if ("uks" in client.query_keywords()[dft_keyword].values.values() or "uhf" in client.query_keywords()[dft_keyword].values.values()) and dft_program == "psi4":
            dft_geom_functionals = {
                key: ["U" + item for item in value]
                for key, value in dft_geom_functionals.items()
            }
    except TypeError:
        pass

    best_opt_lot, rmsd_df = compare_all_rmsd(dft_geom_functionals, odset_dict, ref_geom_fmols)

    padded_log(logger, "BENCHMARK RESULSTS")
    log_dataframe_averages(logger, rmsd_df)

    folder_path_json = res_folder / "json_data"
    folder_path_json.mkdir(parents=True, exist_ok=True)

    rmsd_df.to_json(str(folder_path_json / "results_geom_benchmark.json"))
    logger.info(f"\nDataFrame successfully saved to {folder_path_json}/results_geom_benchmark.json\n")

    folder_path_plots = res_folder / "plots"
    folder_path_plots.mkdir(parents=True, exist_ok=True)

    rmsd_histograms(rmsd_df, smol_name, str(folder_path_plots))

    padded_log(
        logger,
        "Geometry Benchmark finished successfully! Hasta pronto!",
        padding_char=mia0911,
    )

    logger.removeHandler(file_handler)
    file_handler.close()
