import argparse
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from tqdm import tqdm
from typing import Union, List, Tuple, Dict
from pathlib import Path
import qcfractal.interface as ptl
import qcelemental as qcel
import sys, os


# Local application/library specific imports
from beep.utils.logging_utils import *
from beep.utils.plotting_utils import zpve_plot
from beep.zpve import get_zpve_mol


warnings.filterwarnings('ignore')

bcheck = "\u2714"
mia0911 = "\u2606"
gear = "\u2699"

welcome_msg = """       
·······················································································
:                                                                                     :
:  ██████╗ ██╗███╗   ██╗██████╗ ██╗███╗   ██╗ ██████╗                                 :
:  ██╔══██╗██║████╗  ██║██╔══██╗██║████╗  ██║██╔════╝                                 :
:  ██████╔╝██║██╔██╗ ██║██║  ██║██║██╔██╗ ██║██║  ███╗                                :
:  ██╔══██╗██║██║╚██╗██║██║  ██║██║██║╚██╗██║██║   ██║                                :
:  ██████╔╝██║██║ ╚████║██████╔╝██║██║ ╚████║╚██████╔╝                                :
:  ╚═════╝ ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝                                 :
:                                                                                     :
:  ███████╗███╗   ██╗███████╗██████╗  ██████╗ ██╗   ██╗                               :
:  ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔════╝ ╚██╗ ██╔╝                               :
:  █████╗  ██╔██╗ ██║█████╗  ██████╔╝██║  ███╗ ╚████╔╝                                :
:  ██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██║   ██║  ╚██╔╝                                 :
:  ███████╗██║ ╚████║███████╗██║  ██║╚██████╔╝   ██║                                  :
:  ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝                                  :
:                                                                                     :
:  ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗  :
:  ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║  :
:  █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║  :
:  ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║  :
:  ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║  :
:  ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝  :
:                                                                                     :
:  ██████╗ ██╗      █████╗ ████████╗███████╗ ██████╗ ██████╗ ███╗   ███╗              :
:  ██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║              :
:  ██████╔╝██║     ███████║   ██║   █████╗  ██║   ██║██████╔╝██╔████╔██║              :
:  ██╔═══╝ ██║     ██╔══██║   ██║   ██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║              :
:  ██║     ███████╗██║  ██║   ██║   ██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║              :
:  ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝              :
:                                                                                     :
·······················································································

---------------------------------------------------------------------------------------
Welcome to the BEEP  binding energy data extraction workflow!
---------------------------------------------------------------------------------------

"And now I see. With eye serene. The very. Pulse. Of the machine."

                                                ~ Michael Swanwick


Scrutinizing, Leveraging, and Magnifying.


                           By: Stefan Vogt-Geisse

---------------------------------------------------------------------------------------

"""


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="BEEP Binding Energy Evaluation Platform")

    parser.add_argument('--opt-method', type=str, default='mpwb1k-d3bj_def2-tzvp', help='Optimization method')
    parser.add_argument('--be-methods', type=str, nargs='+', default=['WB97X-V', 'M06-HF', 'WPBE-D3MBJ'], help='Binding energy methods')
    parser.add_argument('--server-address', type=str, default='https://152.74.10.245:7777', help='Server address')
    parser.add_argument('--mol-coll-name', type=str, default='carbonyl_hydroxyl_beep-1', help='Molecule collection name')
    parser.add_argument('--surface-model', type=str, help='Surface model to use')
    parser.add_argument('--hessian-clusters', type=str, nargs='*', default=[], help='List of clusters on which the Hessian was computed')
    parser.add_argument('--molecules', type=str, nargs='*', default=[], help='List of molecules')
    parser.add_argument('--be-range', type=float, nargs=2, default=[-0.1, -25.0], help='Binding energy range to filter structures (default: [-0.1, -25.0] kcal/mol)')
    parser.add_argument('--scale-factor', type=float, default=0.958, help='Scaling factor for ZPVE corrections (default: 0.958)')
    parser.add_argument('--exclude-clusters', type=str, nargs='*', default=[], help='List of clusters to exclude from processing')

    return parser.parse_args()


def concatenate_frames(
    client: ptl.FractalClient,
    mol: str,
    ds_w: pd.DataFrame,
    opt_method: str,
    be_range: Tuple[float, float] = (-0.1, -25.0),
    exclude_clusters: List[str] = []
) -> Tuple[pd.DataFrame, bool]:
    """
    Concatenates dataframes containing binding energy values from multiple reaction datasets and filters them
    based on a user-defined binding energy range. Optionally excludes specified clusters.

    Parameters
    ----------
    client : ptl.FractalClient
        Fractal client for dataset access.
    mol : str
        Molecule identifier.
    ds_w : pd.DataFrame
        Dataset containing indices of water data.
    opt_method : str
        Optimization method.
    be_range : Tuple[float, float], optional
        Range for filtering binding energies (BEs). Only structures with BEs within this range will be retained.
        Defaults to (-0.1, -25.0) kcal/mol.
    exclude_clusters : List[str], optional
        List of clusters to exclude from processing. Defaults to an empty list.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        Concatenated dataframe of binding energies and a success flag.
    """
    logger = logging.getLogger("beep")
    df_be = pd.DataFrame()
    method = opt_method.split('_')[0]

    logger.info("Joining the energies of the different clusters.")

    for w in ds_w.df.index:
        # Skip clusters in the exclude list
        if w in exclude_clusters:
            logger.info(f"Skipping excluded cluster: {w}")
            continue

        name_be = f"be_{mol}_{w}_{method}"
        try:
            ds_be = client.get_collection("ReactionDataset", name_be)
        except KeyError:
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
        df = df.reset_index().rename(columns={'index': 'OriginalIndex'})
        df_be = pd.concat([df_be, df.dropna(axis=1, how='all')], axis=0, ignore_index=True)
        logger.info(f"Successfully added collection {name_be}")

    if df_be.empty:
        return df_be, False

    df_be.set_index('OriginalIndex', inplace=True)

    # Identify columns to drop (Without D3BJ)
    cols_to_drop = []
    for col in df_be.columns:
        me, ba = col.split('/')
        for suffix in ['-D3BJ', '-D3MBJ']:
            d3bj_col = f"{me}{suffix}/{ba}"
            if suffix not in me and d3bj_col in df_be.columns:
                cols_to_drop.append(col)
                break

    # Drop the non D3 columns
    logger.info("Deleting columns without dispersion correction")
    df_be.drop(columns=cols_to_drop, inplace=True)

    logger.info("Computing mean values and standard deviation...")
    df_be['Mean_Eb_all_dft'] = df_be.mean(axis=1)
    df_be['StdDev_all_dft'] = df_be.std(axis=1)

    # Filter structures based on the user-specified binding energy range
    logger.info(f"Applying binding energy range of {be_range} kcal/mol")
    df_be = df_be[(df_be['Mean_Eb_all_dft'] >= be_range[1]) & (df_be['Mean_Eb_all_dft'] <= be_range[0])]

    return df_be, not df_be.empty


def zpve_correction(
    name_be: List[str],
    be_methods: List[str],
    lot_opt: str,
    client: ptl.FractalClient,
    scale_factor: float = 1.0,
    basis: str = 'def2-tzvp',
    be_range: Tuple[float, float] = (-0.1, -25.0)
) -> Tuple[pd.DataFrame, Dict[str, List[float]], List[str]]:
    """
    Processes data to apply Zero-Point Vibrational Energy (ZPVE) corrections to binding energies and performs a linear fit.

    Parameters
    ----------
    name_be : List[str]
        List of dataset names.
    be_methods : List[str]
        List of binding energy methods.
    lot_opt : str
        Level of theory and basis set, e.g., 'HF_6-31G'.
    client : ptl.FractalClient
        Fractal client for dataset access.
    scale_factor : float, optional
        Scaling factor for ZPVE corrections, by default 1.0.
    basis : str, optional
        Basis set for the calculation, by default 'def2-tzvp'.
    be_range : Tuple[float, float], optional
        Range for filtering binding energies (BEs). Only structures with BEs within this range will be retained.
        Defaults to (-0.1, -25.0) kcal/mol.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ZPVE corrected binding energies, with 'Delta_ZPVE' as the last column.
    Dict[str, List[float]]
        Dictionary containing the linear fit parameters (slope, intercept, and R-squared) for each binding energy method.
    List[str]
        List of structures that were excluded due to missing or problematic ZPVE corrections.
    """
    logger = logging.getLogger("beep")
    entry_list, df_nocp, df_be, fitting_params = [], pd.DataFrame(), pd.DataFrame(), {}

    logger.info("Starting ZPVE correction procedure")
    for name in name_be:
        # Retrieve entries and binding energies
        ds_be = client.get_collection("ReactionDataset", name)
        entry_list.extend(ds_be.get_index())
        df_be = df_be.append(ds_be.get_values(), ignore_index=False)
        logger.info(f"Extracting and saving binding energies from {name} for ZPVE correction")

        temp_df = ds_be.get_entries()
        df_nocp = df_nocp.append(temp_df[temp_df['stoichiometry'] == 'be_nocp'], ignore_index=False)

    logger.info("Obtaining the ZPVE correction from the harmonic vibrational analysis")
    zpve_corr_dict, todelete = {}, []
    logger.info(f"Extracting Hessian for the following structures:")

    for entry in entry_list:
        logger.info(f"Processing structure {entry}")
        mol_list = df_nocp[df_nocp['name'] == entry]['molecule'].tolist()

        # Extracting ZPVE for the three molecules
        d, d_bol = get_zpve_mol(client, mol_list[0], lot_opt)
        m1, _ = get_zpve_mol(client, mol_list[1], lot_opt, on_imaginary='raise')
        m2, _ = get_zpve_mol(client, mol_list[2], lot_opt, on_imaginary='raise')

        if not (m1 and m2):
            logger.info(f"Molecules {mol_list[1]} and {mol_list[2]} have no Hessian. Compute them first.")
            raise IndexError

        if not d_bol:
            logger.info(f"Appending structure {entry} to the list for deletion.")
            todelete.append(entry)
            continue

        zpve_corr_dict[entry] = (d - m1 - m2) * qcel.constants.hartree2kcalmol
        logger.info(f"Finished processing structure {entry}, the ZPVE correction is: {zpve_corr_dict[entry]}")

    # DataFrame with ZPVE correction
    df_zpve = pd.DataFrame.from_dict(zpve_corr_dict, orient='index', columns=["Delta_ZPVE"])

    # Filter df_be dataframe
    df_be = df_be.drop(todelete)

    # Check number of Hessians
    if len(df_be) < 5:
        raise ValueError('Too few Hessians to construct a ZPVE linear model. Please compute more Hessians.')
    if 5 <= len(df_be) <= 9:
        logger.info(f"WARNING: Number of Hessians is low and may result in a poor linear model. Proceed with caution.")
    else:
        logger.info(f"Total number of Hessian structures: {len(df_be)}")

    # Apply the scale factor to Delta_ZPVE and calculate Eb_ZPVE
    logger.info(f"Applying scaling factor {scale_factor} to the ZPVE correction")
    for bm in be_methods:
        df_zpve['Delta_ZPVE'] *= scale_factor
        zpve_col_name = f"{bm}/{basis}+ZPVE"

        # Sum with the Delta_ZPVE
        df_be[zpve_col_name] = df_be[f"{bm}/{basis}"] + df_zpve['Delta_ZPVE']

        logger.info(f"Fitting procedure for level of theory {bm} (units: kcal/mol)")
        # Convert columns to NumPy arrays and calculate fit parameters
        x = df_be[f"{bm}/{basis}"].to_numpy(dtype=float)
        y = df_be[zpve_col_name].to_numpy(dtype=float)
        m, b = np.polyfit(x, y, 1)
        r_sq = np.corrcoef(x, y)[0, 1] ** 2
        fitting_params[bm] = [m, b, r_sq]
        logger.info(f"Linear model at the {bm} level of theory: BE+ ΔZPVE = {m} * BE + {b}")
        logger.info(f"Fit quality: R² = {r_sq}")

    # Retain only columns with '+ZPVE' in their names
    df_be = df_be[[col for col in df_be.columns if '+ZPVE' in col]]

    # Compute mean and standard deviation for ZPVE corrected energies
    df_be['Mean_Eb_all_dft'] = df_be.mean(axis=1)
    df_be['StdDev_all_dft'] = df_be.std(axis=1)

    # Filter structures based on the user-specified binding energy range
    logger.info(f"Applying binding energy range of {be_range} kcal/mol")
    df_be = df_be[(df_be['Mean_Eb_all_dft'] >= be_range[1]) & (df_be['Mean_Eb_all_dft'] <= be_range[0])]

    # Join Delta_ZPVE with df_be
    df_be = pd.concat([df_be, df_zpve], axis=1)

    # Reorder columns to ensure Delta_ZPVE is the last column
    columns_order = [col for col in df_be.columns if col != 'Delta_ZPVE'] + ['Delta_ZPVE']
    df_be = df_be[columns_order]

    return df_be, fitting_params, todelete


def apply_lin_models(df_be: pd.DataFrame, meth_fit_dict: Dict[str, List[float]], be_methods: List[str],
                     mol: str) -> pd.DataFrame:
    """
    Applies linear models to ZPVE corrected data and plots results for each binding energy method.

    Parameters
    ----------
    df_be : pd.DataFrame
        Dataframe of binding energies.
    meth_fit_dict : dict
        Dictionary with fitting parameters for each method.
    be_methods : list of str
        Binding energy methods.
    mol : str
        Molecule identifier.
    logger : logging.Logger
        Logger for output messages.

    Returns
    -------
    pd.DataFrame
        Dataframe with ZPVE corrected binding energies and calculated mean and standard deviation.
    """
    logger = logging.getLogger("beep")
    zpve_df = pd.DataFrame()
    basis = 'def2-tzvp'

    padded_log(logger, "Applying linear model to correct for ZPVE", padding_char=gear)
    for method, factors in meth_fit_dict.items():
        m, n, R2 = factors
        column_name = f"{method}/{basis}"

        logger.info(f"Applying linear model to {column_name} BEs")
        if column_name in df_be.columns:
            # Apply linear model and store in the dataframe
            scaled_column_name = f"{column_name}_lin_ZPVE"
            zpve_df[scaled_column_name] = df_be[column_name] * m + n

            # Convert the columns to NumPy arrays for plotting
            x = df_be[column_name].to_numpy(dtype=float)
            y = zpve_df[scaled_column_name].to_numpy(dtype=float)
            
            # Generate the plot for the method
            logger.info(f"Creating BE vs BE + ΔZPVE plot for {column_name} saving as {mol}/zpve_{mol}_{method}.svg")
            fig = zpve_plot(x, y, [m, n, R2])
            fig.savefig(f"{mol}/zpve_{mol}_{method}.svg")
            plt.close(fig)
        else:
            raise KeyError(f"Column {column_name} not present in the BE dataframe")

    zpve_df['Mean_Eb_all_dft'] = zpve_df.mean(axis=1)
    zpve_df['StdDev_all_dft'] = zpve_df.std(axis=1) / np.sqrt(3)
    zpve_df = zpve_df[zpve_df['Mean_Eb_all_dft'] < -0.1]

    #log_dictionary(logger, meth_fit_dict, "ZPVE Fitting models")

    return zpve_df


def calculate_mean_std(df_res: pd.DataFrame, mol: str,  logger: logging.Logger) -> str:
    """
    Calculates the mean and standard deviation of binding energies and logs the results.

    Parameters
    ----------
    df_res : pd.DataFrame
        Dataframe with ZPVE corrected binding energies.
    mol : str
        Molecule identifier.
    final_result : str
        Accumulated final result string for logging.
    logger : logging.Logger
        Logger for output messages.

    Returns
    -------
    str
        Updated log content string.
    """
    mean_row = df_res.mean()
    stddev_values = df_res.loc[~df_res.index.str.startswith(('Mean_', 'StdDev_')), 'StdDev_all_dft'].values
    sem = np.sqrt((1 / len(stddev_values) ** 2) * np.sum(stddev_values ** 2))
    std_row = df_res.std()
    std_row['StdDev_all_dft'] = sem

    df_res.loc[f'Mean_{mol}'] = mean_row
    df_res.loc[f'StdDev_{mol}'] = std_row

    mean_val = df_res.loc[f'Mean_{mol}', 'Mean_Eb_all_dft']
    std_val = df_res.loc[f'StdDev_{mol}', 'StdDev_all_dft']

    return df_res, mean_val, std_val

def main():
    args = parse_arguments()
    logger = setup_logging("results", args.mol_coll_name)
    logger.info(welcome_msg)

    client = ptl.FractalClient(address=args.server_address, verify=False)

    # Fetch datasets
    dset_smol = client.get_collection("OptimizationDataset", args.mol_coll_name)
    ds_w = client.get_collection("OptimizationDataset", args.surface_model)
    all_dfs, final_result = [], ''

    # Use all molecules from dset_smol if none are specified in arguments
    mol_list = args.molecules or dset_smol.df.index

    final_result_nz = ''
    final_result_dz = ''
    final_result_lz = ''

    # Process each molecule
    for mol in mol_list:

        logger.info(f"\nProcessing molecule {mol}")

        # Create the new folder
        res_folder = Path.cwd() / str(mol)
        res_folder.mkdir(exist_ok=True)

        # Generate concatenated data frame for binding energy evaluation
        df_no_zpve, success = concatenate_frames(client, mol, ds_w, args.opt_method, be_range=tuple(args.be_range), exclude_clusters=args.exclude_clusters)
        if not success:
            logger.warning(f"No valid binding energies found for {mol}. Skipping...")
            continue

        log_dataframe(logger, df_no_zpve, f"\nBinding energies without ZPVE correction for {mol}\n")
        df_no_zpve.to_csv(f'{res_folder}/be_no_zpve_{mol}.csv')

        # Process ZPVE data
        name_hess_be = [f"be_{mol}_{cluster}_{args.opt_method.split('_')[0]}" for cluster in args.hessian_clusters]

        # Retrieve and process ZPVE data for each method
        df_zpve, fit_data_dict, imag_todelete = zpve_correction(
            name_hess_be, 
            args.be_methods, 
            args.opt_method, 
            client=client, 
            scale_factor=args.scale_factor,
            be_range=tuple(args.be_range)
        )

        # Apply linear models and plot results
        df_zpve_lin = apply_lin_models(df_no_zpve, fit_data_dict, args.be_methods, mol)

        # Delete structures that have imaginary frequencies
        df_zpve_lin.drop(imag_todelete, inplace=True)
        df_no_zpve.drop(imag_todelete, inplace=True)

        # Calculate and log mean and standard deviation
        res_be_no_zpve, mean, sdev = calculate_mean_std(df_no_zpve, mol, logger)
        res_be_zpve, mean, sdev = calculate_mean_std(df_zpve, mol, logger)
        res_be_lin_zpve, mean, sdev = calculate_mean_std(df_zpve_lin, mol, logger)

        padded_log(logger, "Average binding energy results", padding_char=gear)
        log_dataframe(logger, res_be_no_zpve, f"\nBinding energies without ZPVE correction for {mol}\n")
        log_dataframe(logger, res_be_zpve, f"\nBinding energies with direct ZPVE correction for {mol}\n")
        log_dataframe(logger, res_be_lin_zpve, f"\nBinding energies with linear model ZPVE correction for {mol}\n")

        en_log_mol = ''
        en_log_mol = write_energy_log(res_be_no_zpve, mol, en_log_mol, "(NO ZPVE):")
        en_log_mol = write_energy_log(res_be_zpve, mol, en_log_mol, "(Direct ZPVE):")
        en_log_mol = write_energy_log(res_be_lin_zpve, mol, en_log_mol, "(Linear model ZPVE):")
        logger.info(en_log_mol)

        final_result_nz = write_energy_log(res_be_no_zpve, mol, final_result_nz, "(NO ZPVE):")
        final_result_dz = write_energy_log(res_be_zpve, mol, final_result_dz, "(Direct ZPVE):")
        final_result_lz = write_energy_log(res_be_lin_zpve, mol, final_result_lz, "(Linear model ZPVE):")

        # Save all processed data to a CSV file
        padded_log(logger, "Saving all dataframes to CSV", padding_char=gear)
        res_be_no_zpve.to_csv(f'{res_folder}/be_no_zpve_{mol}.csv')
        res_be_zpve.to_csv(f'{res_folder}/be_zpve_{mol}.csv')
        res_be_lin_zpve.to_csv(f'{res_folder}/be_lin_zpve_{mol}.csv')

    padded_log(logger, "Summary of binding energy results", padding_char=gear)
    logger.info(final_result_nz)
    logger.info(final_result_dz)
    logger.info(final_result_lz)

if __name__ == "__main__":
    main()

