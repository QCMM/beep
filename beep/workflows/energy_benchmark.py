"""Energy benchmark workflow — refactored from workflows/launch_energy_benchmark.py."""
import json
import time
import logging
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import qcportal as ptl
import qcelemental as qcel
from pathlib import Path
from qcportal.client import FractalClient
from qcportal.collections import Dataset, OptimizationDataset, ReactionDataset
from qcelemental.models.molecule import Molecule

from ..models.energy_benchmark import EnergyBenchmarkConfig
from ..core.logging_utils import (
    padded_log, log_formatted_list, log_progress, log_energy_mae, beep_banner,
)
from ..core.dft_functionals import hybrid_gga, lrc, meta_hybrid_gga
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
            mol_name, surf_name, _ = name.split("_")
            surf_record = (
                odset_dict[surf_name.upper()]
                .get_record(name=surf_name.upper(), specification=geom_ref_opt_lot)
            )
            if config.use_initial_reference_geometry:
                surf_mod_mol = surf_record.get_initial_molecule()
            else:
                surf_mod_mol = surf_record.get_final_molecule()
            n_surf = len(surf_mod_mol.symbols)
            n_total = len(fmol.symbols)
            mol_f1, mol_f2 = create_molecular_fragments(fmol, n_total - n_surf)
            cbs_col.add_entry(name, fmol)
            cbs_col.add_entry(name + "_f1", mol_f1)
            cbs_col.add_entry(name + "_f2", mol_f2)
            logger.info(f"Adding molecule and fragments of {name} to {cbs_col.name}")
            cbs_col.save()
        else:
            cbs_col.add_entry(name, fmol)
            logger.info(f"Adding molecule {name} to {cbs_col.name}")
            cbs_col.save()


def add_cc_keywords(cbs_col, mol_mult):
    logger = logging.getLogger("beep")
    if mol_mult == 1:
        logger.info(f"\n\nCreating keywords for closed shell coupled cluster computation")
        kw_dict = {"scf_type": "df", "cc_type": "df", "freeze_core": "true"}
        logger.info(f"Keywords dictionary: {kw_dict}")
        kw_dfit = ptl.models.KeywordSet(values=kw_dict)
    elif mol_mult == 2:
        logger.info(f"\n\nCreating keywords for open shell coupled cluster computation")
        kw_dict = {"reference": "uhf", "freeze_core": "true"}
        logger.info(f"Keywords dictionary: {kw_dict}")
        kw_dfit = ptl.models.KeywordSet(values=kw_dict)

    try:
        cbs_col.add_keywords("df", "psi4", kw_dfit)
        cbs_col.save()
        logger.info("Added keywords to Dataset\n")
    except KeyError:
        logger.info("Keyword already set in Dataset, nothing to add\n")


def compute_all_cbs(cbs_col, cbs_list, mol_mult, tag, res_folder=None):
    all_cbs_ids = []
    logger = logging.getLogger("beep")
    id_str = ""
    dir_path = (res_folder / "cbs_ids") if res_folder else Path("cbs_ids")
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    file_path = dir_path / "cbs_ids.dat"

    for lot in cbs_list:
        method, basis = lot.split("_")[0], lot.split("_")[1]
        logger.info(f"\nSendig computations for {method}/{basis}")

        if "scf" not in lot:
            c = cbs_col.compute(method, basis, tag=tag, keywords="df", program="psi4")
        else:
            c = cbs_col.compute(method, basis, tag=tag, program="psi4")

        id_li = c.submitted + c.existing
        id_str += f"{method}_{basis}: {id_li}\n"
        logger.info(f"Submited {len(c.submitted)} Computation to tag {tag}.")
        if len(c.existing) > 0:
            logger.info(f"{len(c.existing)} have already been computed {bcheck}")

    with file_path.open(mode="w") as file:
        file.write(id_str)

    return all_cbs_ids


def check_dataset_status(dataset, cbs_list, wait_interval=1800):
    logger = logging.getLogger("beep")
    while True:
        status_counts = {
            method: {"COMPLETE": 0, "INCOMPLETE": 0, "ERROR": 0}
            for method in set(lot.split("_")[0] for lot in cbs_list)
        }

        for lot in cbs_list:
            method, basis = lot.split("_")
            if "scf" not in method:
                df = dataset.get_records(method=method, basis=basis, program="psi4", keywords="df")
            else:
                df = dataset.get_records(method=method, basis=basis, program="psi4", keywords=None)

            for index, row in df.iterrows():
                status = row["record"].status.upper()
                if status not in status_counts[method]:
                    continue
                status_counts[method][status] += 1
                if status == "ERROR":
                    raise Exception(f"Error in record {index} with level of theory {lot}")

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
    kwargs = {"method": method, "basis": basis, "program": "Psi4", "keywords": None}
    if "scf" not in method:
        kwargs["keywords"] = "df"
    df_records = ds.get_records(**kwargs)
    df_records.index = df_records.index.str.upper()
    records = df_records.loc[struct.upper()]

    if isinstance(records, pd.DataFrame):
        rec = records.iloc[0].iloc[0]
    else:
        rec = records[0]
    return rec


def get_cbs_energy(ds, struct, cbs_lot_list):
    columns = ["SCF", "MP2", "CCSD", "CCSD(T)"]
    index = ["aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ", "CBS"]
    cbs_lot_df = pd.DataFrame(index=index, columns=columns)

    for lot in cbs_lot_list:
        method, basis = lot.split("_")
        rec = get_energy_record(ds, struct, method, basis)

        if "mp2" in method:
            cbs_lot_df.at[basis, "MP2"] = rec.dict()["extras"]["qcvars"]["MP2 CORRELATION ENERGY"]
        elif "ccsd(t)" in method:
            cbs_lot_df.at[basis, "MP2"] = rec.dict()["extras"]["qcvars"]["MP2 CORRELATION ENERGY"]
            cbs_lot_df.at[basis, "CCSD"] = rec.dict()["extras"]["qcvars"]["CCSD CORRELATION ENERGY"]
            cbs_lot_df.at[basis, "CCSD(T)"] = rec.dict()["extras"]["qcvars"]["CCSD(T) CORRELATION ENERGY"]
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
        mol_name, surf_name, _ = bench_struct.split("_")

        struct_cbs_en = get_cbs_energy(cbs_col, bench_struct, cbs_list)
        mol_cbs_en = get_cbs_energy(cbs_col, mol_name.upper(), cbs_list)
        surf_cbs_en = get_cbs_energy(cbs_col, surf_name.upper(), cbs_list)
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
    try:
        client.delete_collection("ReactionDataset", rdset_name)
    except KeyError:
        pass

    ds_be = ReactionDataset(rdset_name, ds_type="rxn", client=client, default_program="psi4")
    ds_be.save()
    n_entries = 0
    for bench_struct in bchmk_structs:
        for lot in dft_opt_lot:
            logger.info(f"Adding entry for {bench_struct} of {lot} geometry")
            try:
                be_stoich = create_be_stoichiometry(odset_dict, bench_struct, lot)
            except (TypeError, KeyError) as e:
                logger.warning(
                    f"Skipping {bench_struct} at {lot}: could not retrieve final molecule "
                    f"(optimization likely in ERROR status). {e}"
                )
                continue
            bench_entry = f"{bench_struct}_{lot}"
            try:
                ds_be.add_rxn(bench_entry, be_stoich)
                n_entries += 1
            except KeyError:
                continue

    ds_be.save()
    logger.info(f"Created a total of {n_entries} in {rdset_name} {bcheck}")
    return ds_be


def create_be_stoichiometry(odset, bench_struct, lot_geom):
    mol_name, surf_name, _ = bench_struct.split("_")
    bench_mol = (
        odset[mol_name.upper()]
        .get_record(name=mol_name.upper(), specification=lot_geom)
        .get_final_molecule()
    )
    bench_struc_mol = (
        odset[bench_struct]
        .get_record(name=bench_struct, specification=lot_geom)
        .get_final_molecule()
    )
    bench_geom = bench_struc_mol.geometry.flatten()
    bench_symbols = bench_struc_mol.symbols

    surf_mod_mol = (
        odset[surf_name.upper()]
        .get_record(name=surf_name.upper(), specification=lot_geom)
        .get_final_molecule()
    )
    surf_symbols = surf_mod_mol.symbols

    n_surf = len(surf_symbols)
    n_total = len(bench_symbols)
    f_bench_struc_mol = ptl.Molecule(
        symbols=bench_symbols,
        geometry=bench_geom,
        fragments=[
            list(range(0, n_total - n_surf)),
            list(range(n_total - n_surf, n_total)),
        ],
    )

    j4 = f_bench_struc_mol.get_fragment(0)     # Small molecule fragment
    j5 = f_bench_struc_mol.get_fragment(1)     # Surface fragment
    j6 = f_bench_struc_mol.get_fragment(0, 1)  # Small molecule with ghost surface
    j7 = f_bench_struc_mol.get_fragment(1, 0)  # Surface with ghost small molecule

    return {
        "default": [
            (f_bench_struc_mol, 1.0), (j4, 1.0), (j5, 1.0),
            (j7, -1.0), (j6, -1.0),
            (surf_mod_mol, -1.0), (bench_mol, -1.0),
        ],
        "be_nocp": [
            (f_bench_struc_mol, 1.0), (surf_mod_mol, -1.0), (bench_mol, -1.0),
        ],
        "ie": [(f_bench_struc_mol, 1.0), (j7, -1.0), (j6, -1.0)],
        "de": [(surf_mod_mol, -1.0), (bench_mol, -1.0), (j4, 1.0), (j5, 1.0)],
    }


def compute_be_dft_energies_eb(ds_be, all_dft, tag, basis="def2-tzvpd",
                                program="psi4"):
    logger = logging.getLogger("beep")
    stoich_list = ["default", "de", "ie", "be_nocp"]
    logger.info(f"Computing energies for the following stoichiometries: {' '.join(stoich_list)} (default = be)")
    log_formatted_list(logger, all_dft, "Sending DFT energy computations for the following functionals:")

    c_list_sub = []
    c_list_exis = []
    for i, func in enumerate(all_dft):
        c_per_func_sub = []
        c_per_func_exis = []
        for stoich in stoich_list:
            c = ds_be.compute(method=func, basis=basis, program=program, stoich=stoich, tag=tag)
            c_list_sub.extend(list(c)[1][1])
            c_per_func_sub.extend(list(c)[1][1])
            c_list_exis.extend(list(c)[0][1])
            c_per_func_exis.extend(list(c)[0][1])
        logger.info(f"\n{func}: Existing {len(c_per_func_exis)}  Submitted {len(c_per_func_sub)}")
        log_progress(logger, i, len(all_dft))

    logger.info(f"Submitted a total of {len(c_list_sub)} DFT computations. {len(c_list_exis)} are already computed")
    return c_list_sub + c_list_exis


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
        surf_mod = bchmk_struct_name.split("_")[1].upper()
        odset_dict[surf_mod] = qcf.get_collection(client, "OptimizationDataset", surf_dset_name)

    ref_geom_fmols = {}
    for struct_name, odset in odset_dict.items():
        record = odset.get_record(struct_name, specification=geom_ref_opt_lot)
        if config.use_initial_reference_geometry:
            ref_geom_fmols[struct_name] = record.get_initial_molecule()
        else:
            ref_geom_fmols[struct_name] = record.get_final_molecule()

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
    add_cc_keywords(cbs_col, mol_mult)

    compute_all_cbs(cbs_col, cbs_list, mol_mult, tag=config.tag_cbs, res_folder=res_folder)
    check_dataset_status(cbs_col, cbs_list)

    ref_df = get_reference_be_result(bchmk_structs, cbs_col, cbs_list)
    logger.info(f"\nFinsihed the Calculation of the CCSD(T)/CBS reference energies:  {bcheck}\n")

    padded_log(logger, "Initializing DFT Binding Energy Computations")
    logger.info(f"The BE will be computed on the following Geometries: {' '.join(dft_opt_lot)}\n")

    ds_be = create_or_load_reaction_dataset_eb(
        client, smol_name, surf_dset_name, bchmk_structs, dft_opt_lot, odset_dict,
    )

    dft_func = {
        "Hybrid GGA": hybrid_gga(),
        "Long range corrected": lrc(),
        "Meta Hybrid GGA": meta_hybrid_gga(),
    }

    for name, dft_f_list in dft_func.items():
        padded_log(logger, f"Sending computations for {name} functionals with a def2-tzvpd basis")
        dft_ids = compute_be_dft_energies_eb(
            ds_be, dft_f_list, basis="def2-tzvpd", program="psi4", tag=config.tag_be,
        )
        qcf.check_jobs_status(client, dft_ids, logger)

    ds_be._disable_query_limit = True
    ds_be.save()

    padded_log(logger, "Retriving energies from ReactionDataset")
    df_be = ds_be.get_values(stoich="default").dropna(axis=1)
    df_ie = ds_be.get_values(stoich="ie").dropna(axis=1)
    df_de = ds_be.get_values(stoich="de").dropna(axis=1)

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

    plot_violins(df_be_plt, bchmk_structs, smol_name, folder_path_plots, ref_df)
    padded_log(logger, "Generating violin plots")
    plot_density_panels(df_be_ae_plt, bchmk_structs, dft_opt_lot, smol_name, folder_path_plots)
    padded_log(logger, "Generating density plots")
    plot_mean_errors(df_be_ae_plt, bchmk_structs, dft_opt_lot, smol_name, folder_path_plots)
    padded_log(logger, "Generating MAE plots")
    plot_ie_vs_de(df_de_re_plt, df_ie_re_plt, bchmk_structs, dft_opt_lot, smol_name, folder_path_plots)
    padded_log(logger, "Generating IE vs DE plots")

    padded_log(logger, "BINDING ENERGY BENCHMARK RESULTS", padding_char=gear)
    padded_log(logger, "BINDING ENERGY MAE")
    log_energy_mae(logger, df_be_ae)
    padded_log(logger, "INTERACTION ENERGY MAE")
    log_energy_mae(logger, df_ie_ae)
    padded_log(logger, "DEFORMATION ENERGY MAE")
    log_energy_mae(logger, df_de_ae)

    logger.removeHandler(file_handler)
    file_handler.close()
