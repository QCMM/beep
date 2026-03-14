"""Extract BE data workflow — refactored from workflows/launch_extract_be_data.py."""
import json
import logging
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import qcelemental as qcel
from pathlib import Path
from ..models.extract import ExtractConfig
from ..core.logging_utils import (
    padded_log, log_dataframe, write_energy_log, beep_banner,
)
from ..core.be_tools import apply_lin_models, calculate_mean_std
from ..adapters import qcfractal_adapter as qcf
from ..adapters.qcfractal_adapter import FractalClient

warnings.filterwarnings("ignore")

bcheck = "\u2714"
gear = "\u2699"

welcome_msg = beep_banner(
    "Binding Energy Data Extraction",
    quote="And now I see. With eye serene. The very. Pulse. Of the machine.",
    quote_author="Michael Swanwick",
    tagline="Scrutinizing, Leveraging, and Magnifying.",
    authors="Stefan Vogt-Geisse",
)


def concatenate_frames(client, mol, ds_w, opt_method, be_range=(-0.1, -25.0),
                       exclude_clusters=None):
    if exclude_clusters is None:
        exclude_clusters = []
    logger = logging.getLogger("beep")
    df_be = pd.DataFrame()
    parts = opt_method.split("_", 1)
    method = parts[0]
    basis = parts[1] if len(parts) == 2 else None

    logger.info("Joining the energies of the different clusters.")

    for w in ds_w.df.index:
        if w in exclude_clusters:
            logger.info(f"Skipping excluded cluster: {w}")
            continue

        if basis:
            name_be = f"be_{mol}_{w}_{method}_{basis}"
        else:
            name_be = f"be_{mol}_{w}_{method}"
        try:
            ds_be = qcf.get_collection(client, "ReactionDataset", name_be)
        except KeyError:
            # Fallback: try method-only name for old datasets (e.g. hf3c_minix -> hf3c)
            if basis:
                name_be_fallback = f"be_{mol}_{w}_{method}"
                try:
                    ds_be = qcf.get_collection(client, "ReactionDataset", name_be_fallback)
                    logger.info(f"Found legacy dataset {name_be_fallback} (no basis in name)")
                    name_be = name_be_fallback
                except KeyError:
                    logger.info(f"ReactionDataset {name_be} not found for molecule: {mol}")
                    continue
            else:
                logger.info(f"ReactionDataset {name_be} not found for molecule: {mol}")
                continue

        ds_be._disable_query_limit = True

        try:
            df = ds_be.get_values(stoich="default")
        except KeyError:
            logger.info(f"ReactionDataset {name_be} exists but seems to be empty, please check.")
            continue

        all_columns = df_be.columns if not df_be.empty else df.columns
        df = df.reindex(columns=all_columns)
        df = df.reset_index().rename(columns={"index": "OriginalIndex"})
        df_be = pd.concat(
            [df_be, df.dropna(axis=1, how="all")], axis=0, ignore_index=True
        )
        logger.info(f"Successfully added collection {name_be}")

    if df_be.empty:
        return df_be, False

    df_be.set_index("OriginalIndex", inplace=True)

    # Identify columns to drop (Without D3BJ)
    cols_to_drop = []
    for col in df_be.columns:
        me, ba = col.split("/")
        for suffix in ["-D3BJ", "-D3MBJ"]:
            d3bj_col = f"{me}{suffix}/{ba}"
            if suffix not in me and d3bj_col in df_be.columns:
                cols_to_drop.append(col)
                break

    logger.info("Deleting columns without dispersion correction")
    df_be.drop(columns=cols_to_drop, inplace=True)

    logger.info("Computing mean values and standard deviation...")
    df_be["Mean_Eb_all_dft"] = df_be.mean(axis=1)
    df_be["StdDev_all_dft"] = df_be.std(axis=1)

    logger.info(f"Applying binding energy range of {be_range} kcal/mol")
    df_be = df_be[
        (df_be["Mean_Eb_all_dft"] >= be_range[1])
        & (df_be["Mean_Eb_all_dft"] <= be_range[0])
    ]

    return df_be, not df_be.empty


def zpve_correction(name_be, be_methods, lot_opt, basis, client,
                    scale_factor=1.0, be_range=(-0.1, -25.0)):
    logger = logging.getLogger("beep")
    entry_list, df_nocp, df_be, fitting_params = [], pd.DataFrame(), pd.DataFrame(), {}

    logger.info("Starting ZPVE correction procedure")
    for name in name_be:
        ds_be = qcf.get_collection(client, "ReactionDataset", name)
        entry_list.extend(ds_be.get_index())
        df_be = df_be.append(ds_be.get_values(), ignore_index=False)
        logger.info(f"Extracting and saving binding energies from {name} for ZPVE correction")

        temp_df = ds_be.get_entries()
        df_nocp = df_nocp.append(
            temp_df[temp_df["stoichiometry"] == "be_nocp"], ignore_index=False
        )

    logger.info("Obtaining the ZPVE correction from the harmonic vibrational analysis")
    zpve_corr_dict, todelete = {}, []
    logger.info(f"Extracting Hessian for the following structures:")

    for entry in entry_list:
        logger.info(f"Processing structure {entry}")
        mol_list = df_nocp[df_nocp["name"] == entry]["molecule"].tolist()

        d, d_bol = qcf.get_zpve_mol(client, mol_list[0], lot_opt)
        m1, _ = qcf.get_zpve_mol(client, mol_list[1], lot_opt, on_imaginary="raise")
        m2, _ = qcf.get_zpve_mol(client, mol_list[2], lot_opt, on_imaginary="raise")

        if len(qcf.fetch_molecules(client, mol_list[2])[0].symbols) == 1:
            if not (m1):
                logger.info(f"Molecules {mol_list[1]} have no Hessian. Compute them first.")
                raise IndexError
        else:
            if not (m1 and m2):
                logger.info(f"Molecules {mol_list[1]} and {mol_list[2]} have no Hessian. Compute them first.")
                raise IndexError

        if not d_bol:
            logger.info(f"Appending structure {entry} to the list for deletion.")
            todelete.append(entry)
            continue

        if d:
            zpve_corr_dict[entry] = (d - m1 - m2) * qcel.constants.hartree2kcalmol
            logger.info(f"Finished processing structure {entry}, the ZPVE correction is: {zpve_corr_dict[entry]}")
        else:
            logger.info(f"Structure {entry}, has no Hessian yet, wait for completion or send the computation.")

    df_zpve = pd.DataFrame.from_dict(zpve_corr_dict, orient="index", columns=["Delta_ZPVE"])
    df_be = df_be.drop(todelete)

    if len(df_be) < 5:
        raise ValueError("Too few Hessians to construct a ZPVE linear model. Please compute more Hessians.")
    if 5 <= len(df_be) <= 9:
        logger.info(f"WARNING: Number of Hessians is low and may result in a poor linear model. Proceed with caution.")
    else:
        logger.info(f"Total number of Hessian structures: {len(df_be)}")

    logger.info(f"Applying scaling factor {scale_factor} to the ZPVE correction")
    df_zpve["Delta_ZPVE"] *= scale_factor
    for bm in be_methods:
        zpve_col_name = f"{bm}/{basis}+ZPVE"
        df_be[zpve_col_name] = df_be[f"{bm}/{basis}"] + df_zpve["Delta_ZPVE"]

        logger.info(f"Fitting procedure for level of theory {bm} (units: kcal/mol)")
        x_raw = df_be[f"{bm}/{basis}"].to_numpy(dtype=float)
        y_raw = df_be[zpve_col_name].to_numpy(dtype=float)

        mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        x = x_raw[mask]
        y = y_raw[mask]

        m, b = np.polyfit(x, y, 1)
        r_sq = np.corrcoef(x, y)[0, 1] ** 2
        fitting_params[bm] = [m, b, r_sq]
        logger.info(f"Linear model at the {bm} level of theory: BE+ \u0394ZPVE = {m} * BE + {b}")
        logger.info(f"Fit quality: R\u00b2 = {r_sq}")

    # Universal model: fit mean uncorrected BE vs mean ZPVE-corrected BE
    uncorr_cols = [f"{bm}/{basis}" for bm in be_methods]
    zpve_cols = [f"{bm}/{basis}+ZPVE" for bm in be_methods]
    x_mean = df_be[uncorr_cols].mean(axis=1).to_numpy(dtype=float)
    y_mean = df_be[zpve_cols].mean(axis=1).to_numpy(dtype=float)
    mask = ~np.isnan(x_mean) & ~np.isnan(y_mean)
    m, b = np.polyfit(x_mean[mask], y_mean[mask], 1)
    r_sq = np.corrcoef(x_mean[mask], y_mean[mask])[0, 1] ** 2
    fitting_params["Mean"] = [m, b, r_sq]
    logger.info(f"Universal linear model (mean BE): BE+\u0394ZPVE = {m} * BE + {b}")
    logger.info(f"Fit quality: R\u00b2 = {r_sq}")

    df_be = df_be[[col for col in df_be.columns if "+ZPVE" in col]]

    df_be["Mean_Eb_all_dft"] = df_be.mean(axis=1)
    df_be["StdDev_all_dft"] = df_be.std(axis=1)

    logger.info(f"Applying binding energy range of {be_range} kcal/mol")
    df_be = df_be[
        (df_be["Mean_Eb_all_dft"] >= be_range[1])
        & (df_be["Mean_Eb_all_dft"] <= be_range[0])
    ]

    df_be = pd.concat([df_be, df_zpve], axis=1, join='inner')
    columns_order = [col for col in df_be.columns if col != "Delta_ZPVE"] + ["Delta_ZPVE"]
    df_be = df_be[columns_order]

    return df_be, fitting_params, todelete


def run(config: ExtractConfig, client: FractalClient) -> None:
    logger = logging.getLogger("beep")
    logger.info(welcome_msg)

    dset_smol = qcf.get_collection(client, "OptimizationDataset", config.mol_coll_name)
    ds_w = qcf.get_collection(client, "OptimizationDataset", config.surface_model)

    mol_list = config.molecules or list(dset_smol.df.index)

    final_result_nz = ""
    final_result_dz = ""
    final_result_lz = ""

    for mol in mol_list:
        logger.info(f"\nProcessing molecule {mol}")

        res_folder = Path.cwd() / str(mol)
        res_folder.mkdir(exist_ok=True)

        # File logging inside the output folder
        log_file = res_folder / f"beep_extract_{mol}.log"
        file_handler = logging.FileHandler(str(log_file), mode='w')
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

        # Write the welcome banner into the log file
        file_handler.emit(logging.LogRecord(
            "beep", logging.INFO, "", 0, welcome_msg, (), None))

        # Save a copy of the input config
        config_path = res_folder / f"extract_{mol}.json"
        config_path.write_text(json.dumps(config.dict(), indent=4, default=str))

        df_no_zpve, success = concatenate_frames(
            client, mol, ds_w, config.opt_method,
            be_range=tuple(config.be_range),
            exclude_clusters=config.exclude_clusters,
        )
        if not success:
            logger.warning(f"No valid binding energies found for {mol}. Skipping...")
            logger.removeHandler(file_handler)
            file_handler.close()
            continue

        res_be_no_zpve, mean, sdev = calculate_mean_std(df_no_zpve, mol, logger)
        log_dataframe(
            logger, res_be_no_zpve,
            f"\nBinding energies without ZPVE correction for {mol}\n",
        )
        res_be_no_zpve.to_csv(f"{res_folder}/be_no_zpve_{mol}.csv")

        if config.no_zpve:
            logger.info("Skipping ZPVE correction and model fitting due to no_zpve flag.")
            final_result_nz = write_energy_log(res_be_no_zpve, mol, final_result_nz, "(NO ZPVE):")
            logger.removeHandler(file_handler)
            file_handler.close()
            continue

        log_dataframe(
            logger, df_no_zpve,
            f"\nBinding energies without ZPVE correction for {mol}\n",
        )
        df_no_zpve.to_csv(f"{res_folder}/be_no_zpve_{mol}.csv")

        opt_parts = config.opt_method.split("_", 1)
        opt_suffix = "_".join(opt_parts)
        name_hess_be = []
        for cluster in config.hessian_clusters:
            name = f"be_{mol}_{cluster}_{opt_suffix}"
            try:
                qcf.get_collection(client, "ReactionDataset", name)
                name_hess_be.append(name)
            except KeyError:
                # Fallback: try method-only name for legacy datasets
                if len(opt_parts) == 2:
                    name_fallback = f"be_{mol}_{cluster}_{opt_parts[0]}"
                    try:
                        qcf.get_collection(client, "ReactionDataset", name_fallback)
                        logger.info(f"Found legacy hessian dataset {name_fallback}")
                        name_hess_be.append(name_fallback)
                    except KeyError:
                        logger.info(f"Hessian dataset {name} not found, skipping")
                else:
                    logger.info(f"Hessian dataset {name} not found, skipping")

        df_zpve, fit_data_dict, imag_todelete = zpve_correction(
            name_hess_be, config.be_methods, config.opt_method,
            config.basis, client=client,
            scale_factor=config.scale_factor,
            be_range=tuple(config.be_range),
        )

        df_zpve_lin = apply_lin_models(
            df_no_zpve, df_zpve, fit_data_dict,
            config.be_methods, config.basis, mol, tuple(config.be_range),
            generate_plots=config.generate_plots,
        )

        df_zpve_lin.drop(imag_todelete, inplace=True, errors='ignore')
        df_no_zpve.drop(imag_todelete, inplace=True, errors='ignore')

        res_be_no_zpve, mean, sdev = calculate_mean_std(df_no_zpve, mol, logger)
        res_be_zpve, mean, sdev = calculate_mean_std(df_zpve, mol, logger)
        res_be_lin_zpve, mean, sdev = calculate_mean_std(df_zpve_lin, mol, logger)

        padded_log(logger, "Average binding energy results", padding_char=gear)
        log_dataframe(logger, res_be_no_zpve, f"\nBinding energies without ZPVE correction for {mol}\n")
        log_dataframe(logger, res_be_zpve, f"\nBinding energies with direct ZPVE correction for {mol}\n")
        log_dataframe(logger, res_be_lin_zpve, f"\nBinding energies with linear model ZPVE correction for {mol}\n")

        en_log_mol = ""
        en_log_mol = write_energy_log(res_be_no_zpve, mol, en_log_mol, "(NO ZPVE):")
        en_log_mol = write_energy_log(res_be_zpve, mol, en_log_mol, "(Direct ZPVE):")
        en_log_mol = write_energy_log(res_be_lin_zpve, mol, en_log_mol, "(Linear model ZPVE):")
        logger.info(en_log_mol)

        final_result_nz = write_energy_log(res_be_no_zpve, mol, final_result_nz, "(NO ZPVE):")
        final_result_dz = write_energy_log(res_be_zpve, mol, final_result_dz, "(Direct ZPVE):")
        final_result_lz = write_energy_log(res_be_lin_zpve, mol, final_result_lz, "(Linear model ZPVE):")

        padded_log(logger, "Saving all dataframes to CSV", padding_char=gear)
        res_be_no_zpve.to_csv(f"{res_folder}/be_no_zpve_{mol}.csv")
        res_be_zpve.to_csv(f"{res_folder}/be_zpve_{mol}.csv")
        res_be_lin_zpve.to_csv(f"{res_folder}/be_lin_zpve_{mol}.csv")

        # Remove per-molecule file handler
        logger.removeHandler(file_handler)
        file_handler.close()

    padded_log(logger, "Summary of binding energy results", padding_char=gear)
    if config.no_zpve:
        logger.info(final_result_nz)
    else:
        logger.info(final_result_nz)
        logger.info(final_result_dz)
        logger.info(final_result_lz)
