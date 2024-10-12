import argparse
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, List, Tuple, Dict
import qcfractal.interface as ptl
import qcelemental as qcel
import sys

warnings.filterwarnings('ignore')

welcome_msg = """       
---------------------------------------------------------------------------------------
Welcome to the BEEP Binding Energy Evaluation Platform 
---------------------------------------------------------------------------------------
"""
def _vibanal_wfn(hess: np.ndarray = None, irrep: Union[int, str] = None, molecule=None, energy=None, project_trans: bool = True, project_rot: bool = True, molden=False,name=None, lt=None):
    """Function to perform analysis of a hessian or hessian block, specifically...
    calling for and printing vibrational and thermochemical analysis, setting thermochemical variables,
    and writing the vibrec and normal mode files.
    Parameters
    ----------
    wfn
        The wavefunction which had its Hessian computed.
    hess
        Hessian to analyze, if not the hessian in wfn.
        (3*nat, 3*nat) non-mass-weighted Hessian in atomic units, [Eh/a0/a0].
    irrep
        The irrep for which frequencies are calculated. Thermochemical analysis is skipped if this is given,
        as only one symmetry block of the hessian has been computed.
    molecule : :py:class:`~psi4.core.Molecule` or qcdb.Molecule, optional
        The molecule to pull information from, if not the molecule in wfn. Must at least have similar
        geometry to the molecule in wfn.
    project_trans
        Should translations be projected in the harmonic analysis?
    project_rot
        Should rotations be projected in the harmonic analysis?
    Returns
    -------
    vibinfo : dict
        A dictionary of vibrational information. See :py:func:`~psi4.driver.qcdb.vib.harmonic_analysis`
    """

    from psi4.driver import qcdb
    from psi4 import core, geometry



    if hess is None:
        print("no hessian")

    else:
        nmwhess = hess


    m=molecule.to_string('xyz')
    m= '\n'.join(m.split('\n')[2:])+'no_reorient'
    mol = geometry(m)
    geom = np.asarray(mol.geometry())
    symbols = [mol.symbol(at) for at in range(mol.natom())]
    vibrec = {'molecule': mol.to_dict(np_out=False), 'hessian': nmwhess.tolist()}
    m = np.asarray([mol.mass(at) for at in range(mol.natom())])
    irrep_labels = mol.irrep_labels()
    vibinfo, vibtext = qcdb.vib.harmonic_analysis(
        nmwhess, geom, m, None, irrep_labels, dipder=None, project_trans=project_trans, project_rot=project_rot)


    if core.has_option_changed('THERMO', 'ROTATIONAL_SYMMETRY_NUMBER'):
        rsn = core.get_option('THERMO', 'ROTATIONAL_SYMMETRY_NUMBER')
    else:
        rsn = mol.rotational_symmetry_number()

    if irrep is None:
        therminfo, thermtext = qcdb.vib.thermo(
            vibinfo,
            T=core.get_option("THERMO", "T"),  # 298.15 [K]
            P=core.get_option("THERMO", "P"),  # 101325. [Pa]
            multiplicity=mol.multiplicity(),
            molecular_mass=np.sum(m),
            sigma=rsn,
            rotor_type=mol.rotor_type(),
            rot_const=np.asarray(mol.rotational_constants()),
            E0=energy)

        core.set_variable("ZPVE", therminfo['ZPE_corr'].data)  # P::e THERMO
        core.set_variable("THERMAL ENERGY CORRECTION", therminfo['E_corr'].data)  # P::e THERMO
        core.set_variable("ENTHALPY CORRECTION", therminfo['H_corr'].data)  # P::e THERMO
        core.set_variable("GIBBS FREE ENERGY CORRECTION", therminfo['G_corr'].data)  # P::e THERMO

        core.set_variable("ZERO K ENTHALPY", therminfo['ZPE_tot'].data)  # P::e THERMO
        core.set_variable("THERMAL ENERGY", therminfo['E_tot'].data)  # P::e THERMO
        core.set_variable("ENTHALPY", therminfo['H_tot'].data)  # P::e THERMO
        core.set_variable("GIBBS FREE ENERGY", therminfo['G_tot'].data)  # P::e THERMO

    else:
        core.print_out('  Thermochemical analysis skipped for partial frequency calculation.\n')
    return vibinfo, therminfo


def zpve_correction(name_be: List[str], be_methods: List[str], lot_opt: str,
                    client: ptl.FractalClient, scale_factor: float = 1.0) -> Tuple[pd.DataFrame, List[float]]:
    """
    Processes data to apply ZPVE corrections and perform a linear fit.

    Parameters
    ----------
    name_be : list of str
        List of dataset names.
    be_method : str
        Binding energy method.
    lot_opt : str
        Level of theory and basis set.
    client : ptl.FractalClient
        Fractal client for dataset access.
    scale_factor : float, optional
        Scaling factor for ZPVE corrections.

    Returns
    -------
    df_all : pd.DataFrame
        DataFrame with ZPVE corrected binding energies, with Delta_ZPVE as the last column.
    fit_params : list of float
        List containing the slope, intercept, and R-squared of the linear fit.
    """
    entry_list,  df_nocp, df_be, fitting_params = [],  pd.DataFrame(), pd.DataFrame(), {}
    basis = 'def2-tzvp'

    for name in name_be:
        # Retrieve entries and binding energies
        ds_be = client.get_collection("ReactionDataset", name)
        entry_list.extend(ds_be.get_index())
        df_be = df_be.append(ds_be.get_values(), ignore_index=False)


        temp_df = ds_be.get_entries()
        df_nocp = df_nocp.append(temp_df[temp_df['stoichiometry'] == 'be_nocp'], ignore_index=False)


    zpve_corr_dict, todelete = {}, []
    for entry in entry_list:
        zpve_list, imaginary = [], []
        mol_list = df_nocp[df_nocp['name'] == entry]['molecule']

        for mol in mol_list:
            try:
                result = client.query_results(driver='hessian', molecule=mol, method=lot_opt.split("_")[0],
                                              basis=lot_opt.split("_")[1])[0]
            except IndexError:
                continue
            hess = result.dict()['return_result']
            energy = result.dict()['extras']['qcvars']['CURRENT ENERGY']
            vib, therm = _vibanal_wfn(hess=hess, molecule=result.get_molecule(), energy=energy)

            if any(abs(num.imag) > 1 for num in vib['omega'].data):
                imaginary.append(True)
            zpve_list.append(therm['ZPE_vib'].data)

        if len(zpve_list) != 3 or any(imaginary):
            todelete.append(entry)
            continue

        d, m1, m2 = zpve_list
        zpve_corr_dict[entry] = (d - m1 - m2) * qcel.constants.hartree2kcalmol


    if lot_opt.split("_")[0] == 'hf3c':
        scale_factor = 0.86

    # Dataframe with ZPVE correction

    df_zpve = pd.DataFrame.from_dict(zpve_corr_dict, orient='index', columns=["Delta_ZPVE"])

    # Filter df_be dataframe
    df_be = df_be.drop(todelete)

    # Apply the scale factor to Delta_ZPVE and calculate Eb_ZPVE
    for bm in be_methods:
        df_zpve['Delta_ZPVE'] *= scale_factor
        zpve_col_name = bm + '/'+basis+'+ZPVE' 

        #Sum with the Delta_ZPVE from df_zpve instead of df_all
        df_be[zpve_col_name] = df_be[bm + '/' + basis] + df_zpve['Delta_ZPVE']

        # Convert columns to NumPy arrays and calculate fit parameters
        x = df_be[bm + '/' + basis].to_numpy(dtype=float)
        y = df_be[zpve_col_name].to_numpy(dtype=float)
        m, b = np.polyfit(x, y, 1)
        r_sq = np.corrcoef(x, y)[0, 1] ** 2
        fitting_params[bm] = [m, b, r_sq]

    # Retain only columns with '+ZPVE' in their names
    df_be = df_be[[col for col in df_be.columns if '+ZPVE' in col]]


    
    # Compute main 
    df_be['Mean_Eb_all_dft'] = df_be.mean(axis=1)
    df_be['StdDev_all_dft'] = df_be.std(axis=1) #/ np.sqrt(3)

    # Filter structures
    df_be = df_be[(df_be['Mean_Eb_all_dft'] > -20.0) & (df_be['Mean_Eb_all_dft'] < -0.2)]

    # Join Delta ZPVE with df_be
    df_be = pd.concat([df_be, df_zpve], axis=1)

    # Reorder columns to ensure Delta_ZPVE is the last column
    columns_order = [col for col in df_be.columns if col != 'Delta_ZPVE'] + ['Delta_ZPVE']
    df_be = df_be[columns_order]

    return df_be, fitting_params



def zpve_plot(x: np.ndarray, y: np.ndarray, fit_params: List[float]) -> plt.Figure:
    """
    Creates a plot of ZPVE corrected binding energies with a linear fit.

    Parameters
    ----------
    x : np.ndarray
        Array of x-values (original binding energies).
    y : np.ndarray
        Array of y-values (ZPVE corrected binding energies).
    fit_params : list of float
        List containing the slope, intercept, and R-squared of the linear fit.

    Returns
    -------
    fig : matplotlib.Figure
        Plot of ZPVE corrected binding energies.
    """
    m, b, r_sq = fit_params
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(x, y, 's', markersize=13)
    ax.plot(x, m * x + b, '--k', label=f'y = {m:.3g}x + {b:.3g}\n$R^2$ = {r_sq:.2g}')
    ax.set_xlabel('$E_b$ / kcal mol$^{-1}$', size=22)
    ax.set_ylabel('$E_b$ + $\Delta$ZPVE / kcal mol$^{-1}$', size=22)
    ax.legend(prop={'size': 20}, loc=2)

    return fig


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="BEEP Binding Energy Evaluation Platform")
    parser.add_argument('--opt_method', type=str, default='mpwb1k-d3bj_def2-tzvp', help='Optimization method')
    parser.add_argument('--be_methods', type=str, nargs='+', default=['WB97X-V', 'M06-HF', 'WPBE-D3MBJ'], help='Binding energy methods')
    parser.add_argument('--server_address', type=str, default='https://152.74.10.245:7777', help='Server address')
    parser.add_argument('--mol_coll_name', type=str, default='carbonyl_hydroxyl_beep-1', help='Molecule collection name')
    parser.add_argument('--surface_model', type=str, help='Surface model to use')
    parser.add_argument('--hessian_clusters', type=str, nargs='*', default=[], help='List of clusters on which the Hessian was computed')
    parser.add_argument('--molecules', type=str, nargs='*', default=[], help='List of molecules')

    return parser.parse_args()

def setup_logging(prefix: str, molecule_name: str) -> logging.Logger:
    """
    Sets up logging for the script.

    Parameters
    ----------
    prefix : str
        Prefix for log file names.
    molecule_name : str
        Name of the molecule being processed.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger("beep")
    logger.setLevel(logging.INFO)
    log_file = f"{prefix}_{molecule_name}.log"
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def log_dataframe(logger, df, title="DataFrame"):
    # Convert the DataFrame to a string with pretty formatting
    df_string = df.to_string()

    # Log the DataFrame with the provided title
    logger.info(f"{title}:\n{df_string}")

def log_dictionary(logger, dictionary, title="Dictionary"):
    # Log the dictionary with the provided title
    logger.info(f"{title}:")
    for key, value in dictionary.items():
        logger.info(f"{key}: m = {value[0]}, n = {value[1]}, R^2 = {value[2]}")


def write_energy_log(df: pd.DataFrame, mol: str, existing_content: str = "") -> str:
    """
    Writes energy values from the DataFrame to a log string.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with energy values.
    mol : str
        Molecule name.
    existing_content : str, optional
        Existing log content to append to.

    Returns
    -------
    str
        Updated log content string.
    """
    content = existing_content
    if not content:
        content += f"{'':<20}{'BE Average':<50}{'BE Standard Deviation'}\n"

    last_two_rows = df.iloc[-2:]
    content += f"{mol:<20}"
    for _, row in last_two_rows.iterrows():
        mean_values = []
        for unit in ['kcal/mol', 'K']:
            mean_val = round(qcel.constants.conversion_factor('kcal/mol', unit) * row['Mean_Eb_all_dft'], 2)
            std_val = round(qcel.constants.conversion_factor('kcal/mol', unit) * row['StdDev_all_dft'], 2)
            mean_values.append(f"{mean_val:.2f} ± {std_val:.2f} [{unit}]")
        content += f"{' '.join(mean_values):<50}"
    content += "\n"
    return content


def concatenate_frames(client: ptl.FractalClient, mol: str, ds_w: pd.DataFrame, opt_method: str) -> Tuple[pd.DataFrame, bool]:
    """
    Concatenates dataframes containing binding energy values from multiple reaction datasets.

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

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        Concatenated dataframe of binding energies and a success flag.
    """
    logger = logging.getLogger("beep")
    df_be = pd.DataFrame()
    method = opt_method.split('_')[0]

    for w in ds_w.df.index:
        if "W7_03" in w:
            continue

        name_be = f"be_{mol}_{w}_{method}"
        try:
            ds_be = client.get_collection("ReactionDataset", name_be)
        except KeyError:
            logger.info(f"ReactionDataset {name_be} not found for molecule: {mol}")
            return None, False

        ds_be._disable_query_limit = True
        df = ds_be.get_values(stoich="default")

        all_columns = df_be.columns if not df_be.empty else df.columns
        df = df.reindex(columns=all_columns)
        df = df.reset_index().rename(columns={'index': 'OriginalIndex'})
        df_be = pd.concat([df_be, df.dropna(axis=1, how='all')], axis=0, ignore_index=True)

    df_be.set_index('OriginalIndex', inplace=True)

    df_be['Mean_Eb_all_dft'] = df_be.mean(axis=1)
    df_be['StdDev_all_dft'] = df_be.std(axis=1) / np.sqrt(3)
    df_be = df_be[(df_be['Mean_Eb_all_dft'] > -20.0) & (df_be['Mean_Eb_all_dft'] < -0.2)]
    #log_dataframe(logger, df_be, f"Raw Binding Energies for {mol}")
    return df_be


def process_zpve(client: ptl.FractalClient, mol: str, be_methods: List[str], hess_list: List[str], opt_method: str,
                 scale_factor: float = 0.958) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    Processes Hessian data and applies ZPVE corrections.

    Parameters
    ----------
    client : ptl.FractalClient
        Fractal client for dataset access.
    mol : str
        Molecule identifier.
    be_methods : list of str
        Binding energy methods.
    hess_list : list of str
        List of Hessian clusters.
    opt_method : str
        Optimization method.
    scale_factor : float, optional
        Scaling factor for ZPVE corrections, default is 0.958.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, List[float]]]
        Dataframe with ZPVE corrected binding energies and a dictionary with fitting parameters.
    """
    meth_fit_dict = {}
    name_hess_be = [f"be_{mol}_{cluster}_{opt_method.split('_')[0]}" for cluster in hess_list]

    # Retrieve and process ZPVE data for each method
    df, meth_fit_dict  = zpve_correction(name_hess_be, be_methods, opt_method, client=client, scale_factor=scale_factor)

    return df, meth_fit_dict


def apply_lin_models(df_be: pd.DataFrame, meth_fit_dict: Dict[str, List[float]], be_methods: List[str],
                     mol: str, logger: logging.Logger) -> pd.DataFrame:
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
    zpve_df = pd.DataFrame()
    
    for method, factors in meth_fit_dict.items():
        m, n, R2 = factors
        column_name = f"{method}/def2-tzvp"

        if column_name in df_be.columns:
            # Apply linear model and store in the dataframe
            scaled_column_name = f"Eb_ZPVE_{column_name}"
            zpve_df[scaled_column_name] = df_be[column_name] * m + n

            # Convert the columns to NumPy arrays for plotting
            x = df_be[column_name].to_numpy(dtype=float)
            y = zpve_df[scaled_column_name].to_numpy(dtype=float)
            
            # Generate the plot for the method
            fig = zpve_plot(x, y, [m, n, R2])
            fig.savefig(f"zpve_{mol}_{method}.svg")
            plt.close(fig)

    zpve_df['Mean_Eb_all_dft'] = zpve_df.mean(axis=1)
    zpve_df['StdDev_all_dft'] = zpve_df.std(axis=1) / np.sqrt(3)
    zpve_df = zpve_df[(zpve_df['Mean_Eb_all_dft'] > -20.0) & (zpve_df['Mean_Eb_all_dft'] < -0.2)]

    logger.info("Hessian data retrieved")
    log_dictionary(logger, meth_fit_dict, "ZPVE Fitting models")

    return zpve_df


def calculate_mean_std(df_res: pd.DataFrame, mol: str, final_result: str, logger: logging.Logger) -> str:
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

    mean_value = df_res.loc[f'Mean_{mol}', 'Mean_Eb_all_dft']
    std_value = df_res.loc[f'StdDev_{mol}', 'StdDev_all_dft']

    logger.info(f"Binding energy for {mol} is {mean_value:.4f} ± {std_value:.4f}")
    #log_dataframe(logger, df_res, f"ZPVE Corrected binding energies for {mol}")
    en_log_mol = write_energy_log(df_res, mol, "")
    final_result = write_energy_log(df_res, mol, final_result)
    logger.info(en_log_mol)
    return df_res, final_result

def main():
    args = parse_arguments()
    logger = setup_logging("results", args.mol_coll_name)
    logger.info("Welcome to the BEEP Binding Energy Evaluation Platform")

    client = ptl.FractalClient(address=args.server_address, verify=False)

    # Fetch datasets
    dset_smol = client.get_collection("OptimizationDataset", args.mol_coll_name)
    ds_w = client.get_collection("OptimizationDataset", args.surface_model)
    all_dfs, final_result = [], ''

    # Use all molecules from dset_smol if none are specified in arguments
    mol_list = args.molecules or dset_smol.df.index

    # Process each molecule
    for mol in mol_list:
        logger.info(f"\nProcessing molecule {mol}")

        # Generate concatenated data frame for binding energy evaluation
        df_no_zpve = concatenate_frames(client, mol, ds_w, args.opt_method)

        # Process ZPVE data
        df_zpve, fit_data_dict = process_zpve(client, mol, args.be_methods, args.hessian_clusters, args.opt_method)

        # Apply linear models and plot results
        df_zpve_lin = apply_lin_models(df_no_zpve, fit_data_dict, args.be_methods, mol, logger)

        # Calculate and log mean and standard deviation if ZPVE data is available
        res_be_no_zpve, res_string_1 = calculate_mean_std(df_no_zpve, mol, final_result, logger)
        res_be_zpve, res_string_2    = calculate_mean_std(df_zpve, mol, final_result, logger)
        res_be_lin_zpve, res_string_3  = calculate_mean_std(df_zpve, mol, final_result, logger)

        # Save all processed data to a CSV file
        res_be_zpve.to_csv(f'be_no_zpve_{mol}.csv')
        res_be_no_zpve.to_csv(f'be_no_zpve_{mol}.csv')
        res_be_lin_zpve.to_csv(f'be_no_zpve_{mol}.csv')

if __name__ == "__main__":
    main()

