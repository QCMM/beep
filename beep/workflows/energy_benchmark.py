"""Energy benchmark workflow — refactored from workflows/launch_energy_benchmark.py."""
import json
import time
import logging
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import qcelemental as qcel
from pathlib import Path

from ..models.energy_benchmark import EnergyBenchmarkConfig
from ..core.logging_utils import (
    padded_log, log_formatted_list, log_progress, log_energy_mae, beep_banner,
)
from ..core.stoichiometry import be_stoichiometry
from ..core.dft_functionals import gga, meta_gga, hybrid_gga, lrc, meta_hybrid_gga
from ..core.plotting_utils import (
    plot_violins, plot_density_panels, plot_mean_errors, plot_ie_vs_de,
)
from ..core.cbs_extrapolation import (
    scf_xtpl_helgaker_3, scf_xtpl_helgaker_2, corl_xtpl_helgaker_2,
)
from ..core.benchmark_utils import (
    create_benchmark_dataset_dict, create_molecular_fragments, get_errors_dataframe,
)
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient, Dataset, ReactionDataset

warnings.filterwarnings("ignore")

bcheck = "\u2714"
mia0911 = "\u2606"
gear = "\u2699"
wstar = "\u2606"

welcome_msg = beep_banner(
    "Binding Energy Benchmark",
    tagline="Steadfastness, Learning, and Mastery.",
    authors="Stefan Vogt-Geisse",
)


def populate_dataset_with_structures(cbs_col, ref_geom_fmols, bchmk_structs,
                                      odset_dict, geom_ref_opt_lot, config=None):
    logger = logging.getLogger("beep")
    for name, fmol in ref_geom_fmols.items():
        if name in bchmk_structs:
            # Extract surface model name: 'H2O_CD1_01_0001' → dataset 'H2O_CD1_01' → surf 'CD1_01'
            dataset_name = name.rsplit("_", 1)[0]
            mol_name = dataset_name.split("_")[0]
            surf_name = dataset_name.split(f"{mol_name}_", 1)[1]
            surf_record = (
                odset_dict[surf_name]
                .get_record(surf_name, geom_ref_opt_lot)
            )
            if config.use_initial_reference_geometry:
                surf_mod_mol = surf_record.initial_molecule
            else:
                surf_mod_mol = surf_record.final_molecule
            len_f1 = len(surf_mod_mol.symbols)
            mol_f1, mol_f2 = create_molecular_fragments(fmol, len_f1)
            cbs_col.add_entry(name=name, molecule=fmol)
            cbs_col.add_entry(name=name + "_f1", molecule=mol_f1)
            cbs_col.add_entry(name=name + "_f2", molecule=mol_f2)
            logger.info(f"Adding molecule and fragments of {name} to {cbs_col.name}")
        else:
            cbs_col.add_entry(name=name, molecule=fmol)
            logger.info(f"Adding molecule {name} to {cbs_col.name}")


def get_cc_keywords(mol_mult):
    """Return the density-fitting keywords dict for correlated methods."""
    logger = logging.getLogger("beep")
    if mol_mult == 1:
        logger.info(f"\n\nKeywords for closed shell coupled cluster computation")
        kw = {"scf_type": "df", "cc_type": "df", "freeze_core": "true"}
    elif mol_mult != 1:
        logger.info(f"\n\nKeywords for open shell coupled cluster computation")
        kw = {
            "reference": "uhf",
            "scf_type": "df",
            "cc_type": "df",
            "freeze_core": "true",
            "qc_module": "OCC",
        }
    else:
        kw = {}
    logger.info(f"Keywords dictionary: {kw}\n")
    return kw


def compute_all_cbs(cbs_col, cbs_list, mol_mult, tag, cc_keywords=None,
                    res_folder=None):
    from qcportal.singlepoint.record_models import QCSpecification, SinglepointDriver

    logger = logging.getLogger("beep")
    id_str = ""
    dir_path = (res_folder / "cbs_ids") if res_folder else Path("cbs_ids")
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    file_path = dir_path / "cbs_ids.dat"

    if cc_keywords is None:
        cc_keywords = {}

    for lot in cbs_list:
        method, basis = lot.split("_")[0], lot.split("_")[1]
        logger.info(f"\nSending computations for {method}/{basis}")

        # SCF uses no special keywords; correlated methods use df keywords
        kw = {} if "scf" in lot else cc_keywords
        spec_name = f"{method}_{basis}" if not kw else f"{method}_{basis}_df"

        qc_spec = QCSpecification(
            program="psi4",
            driver=SinglepointDriver.energy,
            method=method,
            basis=basis,
            keywords=kw,
        )
        cbs_col.add_specification(spec_name, qc_spec)

        result = cbs_col.submit(
            specification_names=[spec_name],
            compute_tag=tag,
        )

        id_str += f"{method}_{basis}: submitted={result.n_inserted} existing={result.n_existing}\n"
        logger.info(f"Submitted {result.n_inserted} computations to tag {tag}.")
        if result.n_existing > 0:
            logger.info(f"{result.n_existing} have already been computed {bcheck}")

    with file_path.open(mode="w") as file:
        file.write(id_str)


def check_dataset_status(dataset, cbs_list, wait_interval=1800):
    from beep.adapters.qcfractal_adapter import is_complete, is_incomplete, is_error

    logger = logging.getLogger("beep")
    while True:
        status_counts = {
            method: {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}
            for method in set(lot.split("_")[0] for lot in cbs_list)
        }

        for lot in cbs_list:
            method, basis = lot.split("_")
            spec_name = f"{method}_{basis}" if "scf" in method else f"{method}_{basis}_df"

            for entry_name, sn, record in dataset.iterate_records(
                specification_names=[spec_name],
            ):
                if record is None:
                    continue
                if is_complete(record.status):
                    status_counts[method]["COMPLETE"] += 1
                elif is_incomplete(record.status):
                    status_counts[method]["INCOMPLETE"] += 1
                elif is_error(record.status):
                    status_counts[method]["ERROR"] += 1
                    raise Exception(
                        f"Error in record {entry_name} with level of theory {lot}"
                    )

        logger.info(f"\nChecking status of computations for CCSD(T)\\CBS:\n")
        for method, counts in status_counts.items():
            logger.info(
                f"{method}: {counts['INCOMPLETE']} INCOMPLETE, "
                f"{counts['COMPLETE']} COMPLETE, {counts['ERROR']} ERROR"
            )

        all_complete = all(
            counts["INCOMPLETE"] == 0 and counts["ERROR"] == 0
            for counts in status_counts.values()
        )
        if all_complete:
            logger.info("\nAll records are COMPLETE. Continuing with the execution")
            break
        if any(counts["ERROR"] > 0 for counts in status_counts.values()):
            logger.info("\nThere are records with ERROR. Proceed with caution")
            break

        time.sleep(wait_interval)


def get_energy_record(ds, struct, method, basis):
    spec_name = f"{method}_{basis}" if "scf" in method else f"{method}_{basis}_df"
    record = None
    for name_variant in [struct, struct.upper()]:
        try:
            record = ds.get_record(name_variant, spec_name)
            if record is not None:
                break
        except Exception:
            continue
    if record is None:
        raise KeyError(
            f"No record for entry '{struct}' with specification '{spec_name}'"
        )
    return record


def get_cbs_energy(ds, struct, cbs_lot_list):
    columns = ["SCF", "MP2", "CCSD", "CCSD(T)"]
    index = ["aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ", "CBS"]
    cbs_lot_df = pd.DataFrame(index=index, columns=columns)

    for lot in cbs_lot_list:
        method, basis = lot.split("_")
        rec = get_energy_record(ds, struct, method, basis)
        props = rec.properties

        if "mp2" in method:
            cbs_lot_df.at[basis, "MP2"] = props.get("mp2 correlation energy",
                                                      props.get("mp2_correlation_energy"))
        elif "ccsd(t)" in method:
            cbs_lot_df.at[basis, "MP2"] = props.get("mp2 correlation energy",
                                                      props.get("mp2_correlation_energy"))
            cbs_lot_df.at[basis, "CCSD"] = props.get("ccsd correlation energy",
                                                       props.get("ccsd_correlation_energy"))
            cbs_lot_df.at[basis, "CCSD(T)"] = props.get("ccsd(t) correlation energy",
                                                          props.get("ccsd(t)_correlation_energy"))
        else:
            cbs_lot_df.at[basis, method.upper()] = rec.return_result

    cbs_lot_df["CCSD(T)"] -= cbs_lot_df["CCSD"]
    cbs_lot_df["CCSD"] -= cbs_lot_df["MP2"]

    cbs_lot_df.at["CBS", "SCF"] = scf_xtpl_helgaker_3(
        "scf_dtq_xtpl", 2,
        cbs_lot_df.at["aug-cc-pVDZ", "SCF"], 3,
        cbs_lot_df.at["aug-cc-pVTZ", "SCF"], 4,
        cbs_lot_df.at["aug-cc-pVQZ", "SCF"],
    )
    cbs_lot_df.at["CBS", "MP2"] = corl_xtpl_helgaker_2(
        "mp2_tq", 3,
        cbs_lot_df.at["aug-cc-pVTZ", "MP2"], 4,
        cbs_lot_df.at["aug-cc-pVQZ", "MP2"],
    )
    cbs_lot_df.at["CBS", "CCSD"] = corl_xtpl_helgaker_2(
        "ccsd_dt", 2,
        cbs_lot_df.at["aug-cc-pVDZ", "CCSD"], 3,
        cbs_lot_df.at["aug-cc-pVTZ", "CCSD"],
    )
    cbs_lot_df.at["CBS", "CCSD(T)"] = corl_xtpl_helgaker_2(
        "ccsd(t)_dt", 2,
        cbs_lot_df.at["aug-cc-pVDZ", "CCSD(T)"], 3,
        cbs_lot_df.at["aug-cc-pVTZ", "CCSD(T)"],
    )

    cbs_lot_df["NET"] = cbs_lot_df.sum(axis=1)
    return cbs_lot_df


def get_reference_be_result(bchmk_structs, cbs_col, cbs_list):
    logger = logging.getLogger("beep")
    result_df = pd.DataFrame(columns=["IE", "DE", "BE"])

    for bench_struct in bchmk_structs:
        padded_log(logger, f"Calculating CBS extrapolations for {bench_struct}")
        logger.info("\nInteraction Energy : IE\nDeformation Energy : DE\nBinding Energy : BE\n")
        dataset_name = bench_struct.rsplit("_", 1)[0]
        mol_name = dataset_name.split("_")[0]
        surf_name = dataset_name.split(f"{mol_name}_", 1)[1]

        struct_cbs_en = get_cbs_energy(cbs_col, bench_struct, cbs_list)
        mol_cbs_en = get_cbs_energy(cbs_col, mol_name.upper(), cbs_list)
        surf_cbs_en = get_cbs_energy(cbs_col, surf_name, cbs_list)
        struct_cbs_en_f1 = get_cbs_energy(cbs_col, bench_struct + "_f1", cbs_list)
        struct_cbs_en_f2 = get_cbs_energy(cbs_col, bench_struct + "_f2", cbs_list)

        ie = (struct_cbs_en - (struct_cbs_en_f1 + struct_cbs_en_f2)) * qcel.constants.hartree2kcalmol
        be = (struct_cbs_en - (mol_cbs_en + surf_cbs_en)) * qcel.constants.hartree2kcalmol
        de = ((struct_cbs_en_f1 + struct_cbs_en_f2) - (mol_cbs_en + surf_cbs_en)) * qcel.constants.hartree2kcalmol

        logger.info(f"\nCCSD(T)/CBS result for structure: {bench_struct}")
        df_dict = {"IE": ie, "DE": de, "BE": be}
        for key, df in df_dict.items():
            logger.info(f"\nThe CCSD(T)/CBS incremental table for the {wstar} {key}{wstar} :")
            df_formatted = df.fillna("-")
            logger.info(f"\n{df_formatted.to_string()}\n")

        temp_row = {key: df.loc["CBS", "NET"] for key, df in df_dict.items()}
        result_df = pd.concat(
            [result_df, pd.DataFrame(temp_row, index=[bench_struct])],
            axis=0, join="outer",
        )

    padded_log(logger, "\n FINAL CCSD(T)/CBS RESULTS\n")
    logger.info(result_df)
    return result_df


def create_or_load_reaction_dataset_eb(client, smol_name, surf_dset_name,
                                        bchmk_structs, dft_opt_lot, odset_dict):
    logger = logging.getLogger("beep")
    rdset_name = f"bchmk_be_{smol_name}_{surf_dset_name}"
    logger.info(f"Creating a loading ReactionDataset: {rdset_name}\n")

    # Create one dataset per stoichiometry type
    for stoich_type in qcf.STOICH_TYPES:
        ds_name = f"{rdset_name}_{stoich_type}"
        qcf.create_reaction_dataset(client, ds_name, program="psi4")

    n_entries = 0
    for bench_struct in bchmk_structs:
        for lot in dft_opt_lot:
            logger.info(f"Adding entry for {bench_struct} of {lot} geometry")
            try:
                smol_mol, surf_mol, struc_mol = _fetch_be_molecules(
                    odset_dict, bench_struct, lot
                )
                be_stoich = be_stoichiometry(smol_mol, surf_mol, struc_mol, logger)
            except (TypeError, KeyError) as e:
                logger.warning(
                    f"Skipping {bench_struct} at {lot}: could not retrieve final molecule "
                    f"(optimization likely in ERROR status). {e}"
                )
                continue
            bench_entry = f"{bench_struct}_{lot}"
            try:
                qcf.add_reaction(client, rdset_name, bench_entry, be_stoich)
                n_entries += 1
            except KeyError:
                continue

    logger.info(f"Created a total of {n_entries} in {rdset_name} {bcheck}")
    return rdset_name


def _fetch_be_molecules(odset, bench_struct, lot_geom):
    """Fetch the three molecules needed for BE stoichiometry from optimization datasets.

    Returns (smol_mol, surf_mol, struc_mol) — the small molecule, surface model,
    and full complex, all as optimized geometries.
    """
    dataset_name = bench_struct.rsplit("_", 1)[0]
    mol_name = dataset_name.split("_")[0]
    surf_name = dataset_name.split(f"{mol_name}_", 1)[1]
    smol_mol = (
        odset[mol_name.upper()]
        .get_record(mol_name.upper(), lot_geom)
        .final_molecule
    )
    surf_mol = (
        odset[surf_name]
        .get_record(surf_name, lot_geom)
        .final_molecule
    )
    struc_mol = (
        odset[bench_struct]
        .get_record(bench_struct, lot_geom)
        .final_molecule
    )
    return smol_mol, surf_mol, struc_mol


def compute_be_dft_energies_eb(client, rdset_base_name, all_dft, tag,
                                basis="def2-tzvpd", program="psi4"):
    logger = logging.getLogger("beep")
    logger.info(f"Computing energies for the following stoichiometries: {' '.join(qcf.STOICH_TYPES)} (bsse = be)")
    log_formatted_list(logger, all_dft, "Sending DFT energy computations for the following functionals:")

    total_submitted = 0
    total_existing = 0
    for i, func in enumerate(all_dft):
        func_submitted = 0
        func_existing = 0
        for stoich in qcf.STOICH_TYPES:
            result = qcf.submit_energies(
                client, rdset_base_name,
                method=func, basis=basis, program=program,
                stoich=stoich, tag=tag,
            )
            func_submitted += result.n_inserted
            func_existing += result.n_existing
        total_submitted += func_submitted
        total_existing += func_existing
        logger.info(f"\n{func}: Existing {func_existing}  Submitted {func_submitted}")
        log_progress(logger, i, len(all_dft))

    logger.info(f"Submitted a total of {total_submitted} DFT computations. {total_existing} are already computed")

    # Collect record IDs for monitoring
    record_ids = []
    for stoich in qcf.STOICH_TYPES:
        ds_name = f"{rdset_base_name}_{stoich}"
        ds = client.get_dataset("reaction", ds_name)
        for _, _, record in ds.iterate_records():
            if record is not None:
                record_ids.append(record.id)
    return record_ids


def run(config: EnergyBenchmarkConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")

    smol_name = config.molecule

    # Create output folder: <molecule>/energy_benchmark/
    res_folder = Path.cwd() / smol_name / "energy_benchmark"
    res_folder.mkdir(parents=True, exist_ok=True)

    # File logging inside the output folder
    log_file = res_folder / f"beep_energy_benchmark_{smol_name}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Save a copy of the input config
    config_path = res_folder / f"energy_benchmark_{smol_name}.json"
    config_path.write_text(json.dumps(config.dict(), indent=4, default=str))

    logger.info(welcome_msg)
    gr_method, gr_basis, gr_program = config.reference_be_level_of_theory
    geom_ref_opt_lot = gr_method + "_" + gr_basis

    bchmk_structs = config.benchmark_structures
    surf_dset_name = config.surface_model_collection
    smol_dset_name = config.small_molecule_collection
    dft_opt_lot = config.opt_level_of_theory

    padded_log(logger, "Starting BEEP Energy benchmark procedure", padding_char=wstar)
    logger.info(f"Molecule: {smol_name}")
    logger.info(f"Surface Model: {smol_dset_name}")
    logger.info(f"Benchmark Structures: {' '.join(bchmk_structs)}")
    logger.info(f"DFT and SQM  geometry levels of theory: {' '.join(dft_opt_lot)}")

    smol_dset = qcf.get_collection(client, "OptimizationDataset", smol_dset_name)
    mol_mult = qcf.get_molecular_multiplicity(client, smol_dset, smol_name)
    logger.info(f"\nThe molecular multiplicity of {smol_name} is {mol_mult}\n\n")
    logger.info(f"Retriving data of the reference equilibirum geometries at {gr_method}/{gr_basis}:\n")

    odset_dict = {}
    bchmk_dset_names = create_benchmark_dataset_dict(bchmk_structs)

    qcf.check_collection_existence(client, *bchmk_dset_names.values())
    qcf.check_collection_existence(client, smol_dset_name)
    qcf.check_collection_existence(client, surf_dset_name)

    odset_dict = {smol_name: smol_dset}
    for bchmk_struct_name, odset_name in bchmk_dset_names.items():
        odset_dict[bchmk_struct_name] = qcf.get_collection(client, "OptimizationDataset", odset_name)
        surf_mod = odset_name.split(f"{smol_name}_", 1)[1]
        odset_dict[surf_mod] = qcf.get_collection(client, "OptimizationDataset", surf_dset_name)

    ref_geom_fmols = {}
    for struct_name, odset in odset_dict.items():
        record = odset.get_record(struct_name, geom_ref_opt_lot)
        if config.use_initial_reference_geometry:
            ref_geom_fmols[struct_name] = record.initial_molecule
        else:
            ref_geom_fmols[struct_name] = record.final_molecule

    padded_log(logger, "CCSD(T)/CBS computations:")

    cbs_list = [
        "scf_aug-cc-pVDZ",
        "scf_aug-cc-pVTZ",
        "scf_aug-cc-pVQZ",
        "mp2_aug-cc-pVQZ",
        "ccsd(t)_aug-cc-pVDZ",
        "ccsd(t)_aug-cc-pVTZ",
    ]

    log_formatted_list(logger, cbs_list, "Energies to compute for CCSD(T)/CBS (Semi-enlightend listing) : ")

    logger.info("\nCreating Dataset collection for CCSD(T)/CBS:")
    cbs_col = qcf.get_or_create_collection(
        client, "cbs" + "_" + smol_name + "_" + surf_dset_name, Dataset,
    )

    logger.info(f"\nAdding molecules and fragments to {cbs_col.name} Database collection:\n")
    populate_dataset_with_structures(cbs_col, ref_geom_fmols, bchmk_structs, odset_dict, geom_ref_opt_lot, config)
    cc_kw = get_cc_keywords(mol_mult)

    compute_all_cbs(cbs_col, cbs_list, mol_mult, tag=config.tag_cbs,
                    cc_keywords=cc_kw, res_folder=res_folder)
    check_dataset_status(cbs_col, cbs_list)

    ref_df = get_reference_be_result(bchmk_structs, cbs_col, cbs_list)
    logger.info(f"\nFinsihed the Calculation of the CCSD(T)/CBS reference energies:  {bcheck}\n")

    padded_log(logger, "Initializing DFT Binding Energy Computations")
    logger.info(f"The BE will be computed on the following Geometries: {' '.join(dft_opt_lot)}\n")

    rdset_base = create_or_load_reaction_dataset_eb(
        client, smol_name, surf_dset_name, bchmk_structs, dft_opt_lot, odset_dict,
    )

    dft_func = {
        "GGA": gga(),
        "Meta GGA": meta_gga(),
        "Hybrid GGA": hybrid_gga(),
        "Long range corrected": lrc(),
        "Meta Hybrid GGA": meta_hybrid_gga(),
    }

    if config.custom_dft_functionals:
        dft_func["Custom"] = config.custom_dft_functionals

    for name, dft_f_list in dft_func.items():
        padded_log(logger, f"Sending computations for {name} functionals with a {config.be_basis} basis")
        dft_ids = compute_be_dft_energies_eb(
            client, rdset_base, dft_f_list, basis=config.be_basis, program="psi4", tag=config.tag_be,
        )
        qcf.check_jobs_status(client, dft_ids, logger)

    padded_log(logger, "Retriving energies from ReactionDataset")
    df_be = qcf.fetch_reaction_values(client, rdset_base, stoich="bsse").dropna(axis=1)
    df_ie = qcf.fetch_reaction_values(client, rdset_base, stoich="ie").dropna(axis=1)
    df_de = qcf.fetch_reaction_values(client, rdset_base, stoich="de").dropna(axis=1)

    df_be_ae, df_be_re = get_errors_dataframe(df_be, ref_df["BE"].to_dict())
    df_ie_ae, df_ie_re = get_errors_dataframe(df_ie, ref_df["IE"].to_dict())
    df_de_ae, df_de_re = get_errors_dataframe(df_de, ref_df["DE"].to_dict())

    folder_path_json = res_folder / "json_data"
    folder_path_json.mkdir(parents=True, exist_ok=True)

    padded_log(logger, "Saving BE data in json files")
    df_be.to_json(folder_path_json / "BE_DFT.json", orient="index")
    logger.info(f"Saved BE dataframe in BE_DFT.json. {bcheck}\n")
    df_be_ae.to_json(folder_path_json / "BE_AE_DFT.json", orient="index")
    logger.info(f"Saved BE absolute error  dataframe in BE_AE_DFT.json. {bcheck}\n")
    df_be_re.to_json(folder_path_json / "BE_RE_DFT.json", orient="index")
    logger.info(f"Saved BE relative error  dataframe in BE_AE_DFT.json. {bcheck}\n")

    df_ie.to_json(folder_path_json / "IE_DFT.json", orient="index")
    logger.info(f"Saved IE dataframe in iE_DFT.json. {bcheck}\n")
    df_ie_ae.to_json(folder_path_json / "IE_AE_DFT.json", orient="index")
    logger.info(f"Saved IE absolute error  dataframe in IE_AE_DFT.json. {bcheck}\n")
    df_ie_re.to_json(folder_path_json / "IE_RE_DFT.json", orient="index")
    logger.info(f"Saved IE relative error  dataframe in IE_AE_DFT.json. {bcheck}\n")

    df_de.to_json(folder_path_json / "DE_DFT.json", orient="index")
    logger.info(f"Saved DE dataframe in DE_DFT.json. {bcheck}\n")
    df_de_ae.to_json(folder_path_json / "DE_AE_DFT.json", orient="index")
    logger.info(f"Saved DE absolute error  dataframe in DE_AE_DFT.json. {bcheck}\n")
    df_de_re.to_json(folder_path_json / "DE_RE_DFT.json", orient="index")
    logger.info(f"Saved DE relative error  dataframe in DE_AE_DFT.json. {bcheck}\n")

    padded_log(logger, "Generating BE benchmark plots")

    folder_path_plots = res_folder / "plots"
    folder_path_plots.mkdir(parents=True, exist_ok=True)

    df_be_plt = pd.read_json(folder_path_json / "BE_DFT.json", orient="index")
    df_be_ae_plt = pd.read_json(folder_path_json / "BE_AE_DFT.json", orient="index")
    df_de_re_plt = pd.read_json(folder_path_json / "DE_RE_DFT.json", orient="index")
    df_ie_re_plt = pd.read_json(folder_path_json / "IE_RE_DFT.json", orient="index")

    try:
        plot_violins(df_be_plt, bchmk_structs, smol_name, folder_path_plots, ref_df)
        padded_log(logger, "Generating violin plots")
        plot_density_panels(df_be_ae_plt, bchmk_structs, dft_opt_lot, smol_name, folder_path_plots)
        padded_log(logger, "Generating density plots")
        plot_mean_errors(df_be_ae_plt, bchmk_structs, dft_opt_lot, smol_name, folder_path_plots)
        padded_log(logger, "Generating MAE plots")
        plot_ie_vs_de(df_de_re_plt, df_ie_re_plt, bchmk_structs, dft_opt_lot, smol_name, folder_path_plots)
        padded_log(logger, "Generating IE vs DE plots")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}. Data files are saved, plots can be regenerated.")

    padded_log(logger, "BINDING ENERGY BENCHMARK RESULTS", padding_char=gear)
    padded_log(logger, "BINDING ENERGY MAE")
    log_energy_mae(logger, df_be_ae)
    padded_log(logger, "INTERACTION ENERGY MAE")
    log_energy_mae(logger, df_ie_ae)
    padded_log(logger, "DEFORMATION ENERGY MAE")
    log_energy_mae(logger, df_de_ae)

    logger.removeHandler(file_handler)
    file_handler.close()
