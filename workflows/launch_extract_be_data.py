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
Welcome to the BEEP  data extraction workflow!
---------------------------------------------------------------------------------------

"And now I see. With eye serene. The very. Pulse. Of the machine."

                                                ~ Michael Swanwick


Scrutinizing, Leveraging, and Magnifying.


                           By: Stefan Vogt-Geisse
"""

def suppress_stdout(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save the current stdout
        original_stdout = sys.stdout
        try:
            # Redirect stdout to devnull
            sys.stdout = open(os.devnull, 'w')
            # Call the function with its arguments
            return func(*args, **kwargs)
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
    return wrapper


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

    from psi4.driver.qcdb.vib import harmonic_analysis, thermo
    harmonic = suppress_stdout(harmonic_analysis)
    therm = suppress_stdout(thermo)
    from psi4 import core, geometry

    nmwhess = hess

    m=molecule.to_string('xyz')
    m= '\n'.join(m.split('\n')[2:])+'no_reorient'
    mol = geometry(m)
    geom = np.asarray(mol.geometry())
    symbols = [mol.symbol(at) for at in range(mol.natom())]
    vibrec = {'molecule': mol.to_dict(np_out=False), 'hessian': nmwhess.tolist()}
    m = np.asarray([mol.mass(at) for at in range(mol.natom())])
    irrep_labels = mol.irrep_labels()
    #vibinfo, vibtext = qcdb.vib.harmonic_analysis(
    #    nmwhess, geom, m, None, irrep_labels, dipder=None, project_trans=project_trans, project_rot=project_rot)
    vibinfo, vibtext = harmonic(
        nmwhess, geom, m, None, irrep_labels, dipder=None, project_trans=project_trans, project_rot=project_rot)


    if core.has_option_changed('THERMO', 'ROTATIONAL_SYMMETRY_NUMBER'):
        rsn = core.get_option('THERMO', 'ROTATIONAL_SYMMETRY_NUMBER')
    else:
        rsn = mol.rotational_symmetry_number()

    if irrep is None:
        therminfo, thermtext = therm(
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
    logger.info(f"{title}\n{df_string}")

def log_dictionary(logger, dictionary, title="Dictionary"):
    # Log the dictionary with the provided title
    logger.info(f"{title}:")
    for key, value in dictionary.items():
        logger.info(f"{key}: m = {value[0]}, n = {value[1]}, R^2 = {value[2]}")


def write_energy_log(df: pd.DataFrame, mol: str, existing_content: str = "", comment: str  = "") -> str:
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
        content += f"{'':<50}{'BE Average':<50}{'BE Standard Deviation'}\n"

    last_two_rows = df.iloc[-2:]
    content += f"{mol} {comment:<30}"
    for _, row in last_two_rows.iterrows():
        mean_values = []
        for unit in ['kcal/mol', 'K']:
            mean_val = -1 * round(qcel.constants.conversion_factor('kcal/mol', unit) * row['Mean_Eb_all_dft'], 2)
            std_val =  round(qcel.constants.conversion_factor('kcal/mol', unit) * row['StdDev_all_dft'], 2)
            mean_values.append(f"{mean_val:.2f} ± {std_val:.2f} [{unit}]")
        content += f"{'   '.join(mean_values):<50}"
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

    padded_log(logger, "Joining the energies of the different clusters.", padding_char=mia0911)

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
        logger.info(f"Succesfully added collection {name_be}  {bcheck}")

    df_be.set_index('OriginalIndex', inplace=True)

    # Identify columns to drop (Without D3BJ)
    cols_to_drop = []
    for col in df_be.columns:
        me = col.split('/')[0]
        ba = col.split('/')[1]
        if '-D3BJ' not in me:
            d3bj_col = me  + '-D3BJ/' + ba
            if d3bj_col in df_be.columns:
                cols_to_drop.append(col)
    
    # Drop the columns
    logger.info("Deleting Columns without dispersion correction")
    df_be.drop(columns=cols_to_drop, inplace=True)

    logger.info(f"\n Computing mean values and standart deviation....")
    df_be['Mean_Eb_all_dft'] = df_be.mean(axis=1)
    df_be['StdDev_all_dft'] = df_be.std(axis=1) #/ np.sqrt(3)
    df_be = df_be[df_be['Mean_Eb_all_dft'] < -0.2]
    return df_be


def get_zpve_mol(client: ptl.FractalClient, mol: int, lot_opt: str, on_imaginary: str = 'return') -> Union[bool, Any]:
    """
    Compute vibrational analysis and handle imaginary frequencies.

    Parameters:
    - mol: Molecule object.
    - lot_opt: Level of theory and basis set, e.g., 'HF_6-31G'.
    - on_imaginary: Specifies behavior when imaginary frequencies are found.
        - 'return': Return False.
        - 'raise': Raise a ValueError.

    Returns:
    - The computed zero-point vibrational energy (ZPVE) data, or False if imaginary frequencies are found and on_imaginary='return'.
    """
    logger = logging.getLogger("beep")
    mol_form = client.query_molecules(mol)[0].dict()['identifiers']['molecular_formula']
    method = lot_opt.split("_")[0]
    basis  = lot_opt.split("_")[1]
    if method[0] == 'U':
        method = method[1:]

    try:
        result = client.query_results(
            driver='hessian',
            molecule=mol,
            method=method,
            basis=basis
        )[0]
        logger.info(f"Molecule {mol} with molecular formula {mol_form} has a Hessian {bcheck}")
    except IndexError:
        logger.info(f"Molecule {mol} with molecular formula {mol_form} has not finished computing.")
        return None, True  # Use return instead of continue in a function


    hess = result.dict()['return_result']
    energy = result.dict()['extras']['qcvars']['CURRENT ENERGY']
    vib, therm = _vibanal_wfn(hess=hess, molecule=result.get_molecule(), energy=energy)

    # Check for imaginary frequencies
    imag_freqs = [num for num in vib['omega'].data if abs(num.imag) > 1]
    if imag_freqs:
        if on_imaginary == 'raise':
            raise ValueError(f"There are imaginary frequencies,  {imag_freqs}.  You will need to reoptimize {mol} I am affraid.")
        elif on_imaginary == 'return':
            logger.info(f"There are imaginary frequencies: {imag_freqs} proceed with caution")
            return therm['ZPE_vib'].data, False
        else:
            raise ValueError(f"Invalid option for on_imaginary: {on_imaginary}")

    return therm['ZPE_vib'].data, True  

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
    logger = logging.getLogger("beep")
    entry_list,  df_nocp, df_be, fitting_params = [],  pd.DataFrame(), pd.DataFrame(), {}
    basis = 'def2-tzvp'

    padded_log(logger, "Starting ZPVE correction procedure", padding_char=gear)
    for name in name_be:
        # Retrieve entries and binding energies
        ds_be = client.get_collection("ReactionDataset", name)
        entry_list.extend(ds_be.get_index())
        df_be = df_be.append(ds_be.get_values(), ignore_index=False)
        logger.info(f"Extrating and saving binding energies from {name} for ZPVE correction {bcheck}")


        temp_df = ds_be.get_entries()
        df_nocp = df_nocp.append(temp_df[temp_df['stoichiometry'] == 'be_nocp'], ignore_index=False)


    padded_log(logger, "Obtaining the ZPVE correction from the harmonic vibtrational analysis", padding_char=mia0911)
    zpve_corr_dict, todelete = {}, []
    logger.info(f"\nExtracting Hessian:\n")
    log_formatted_list(logger, entry_list, f"Structures for Hessian Extraction", max_rows = 5)
    todelete = []
    for entry in entry_list:
        logger.info(f"\nProcessing structure {entry}")
        imaginary = []
        mol_list = df_nocp[df_nocp['name'] == entry]['molecule']

        # Extracting ZPVE for the tree molecules
        mol_list = list(mol_list)
        d, d_bol = get_zpve_mol(client, mol_list[0], lot_opt)

        m1, _ = get_zpve_mol(client, mol_list[1], lot_opt, on_imaginary = 'raise')
        m2, _ = get_zpve_mol(client, mol_list[2], lot_opt, on_imaginary = 'raise')


        if not (m1 and m2):
            logger.info("Molecule {mol_list[1]} and {mol_list[2]} have no Hessian. Compute them first.")
            raise IndexError

        if not d_bol:
            logger.info(f"Appending structure {entry} into list to delete.")
            todelete.append(entry)
            continue

        zpve_corr_dict[entry] = (d - m1 - m2) * qcel.constants.hartree2kcalmol
        logger.info(f"Finished processing structure {entry}, the ZPVE correction is: {zpve_corr_dict[entry]}\n\n")

    if lot_opt.split("_")[0] == 'hf3c':
        scale_factor = 0.86

    # Dataframe with ZPVE correction

    df_zpve = pd.DataFrame.from_dict(zpve_corr_dict, orient='index', columns=["Delta_ZPVE"])

    # Filter df_be dataframe
    df_be = df_be.drop(todelete)

    # Apply the scale factor to Delta_ZPVE and calculate Eb_ZPVE
    logger.info(f"Applying scaling factor {scale_factor} to the ZPVE correction\n")
    for bm in be_methods:
        df_zpve['Delta_ZPVE'] *= scale_factor
        zpve_col_name = bm + '/'+basis+'+ZPVE' 

        #Sum with the Delta_ZPVE from df_zpve instead of df_all
        df_be[zpve_col_name] = df_be[bm + '/' + basis] + df_zpve['Delta_ZPVE']

        logger.info(f"Fitting procedure for level of theory {bm} all units are in kcal/mol")
        # Convert columns to NumPy arrays and calculate fit parameters
        x = df_be[bm + '/' + basis].to_numpy(dtype=float)
        y = df_be[zpve_col_name].to_numpy(dtype=float)
        m, b = np.polyfit(x, y, 1)
        r_sq = np.corrcoef(x, y)[0, 1] ** 2
        fitting_params[bm] = [m, b, r_sq]
        logger.info(f"The linear model  at the {bm} level of theory is: BE+ ΔZPVE = {m} BE + {b}")
        logger.info(f"The qualtiy of the fit: R2 = {r_sq}\n")

    # Retain only columns with '+ZPVE' in their names
    df_be = df_be[[col for col in df_be.columns if '+ZPVE' in col]]

    # Compute main 
    df_be['Mean_Eb_all_dft'] = df_be.mean(axis=1)
    df_be['StdDev_all_dft'] = df_be.std(axis=1) #/ np.sqrt(3)

    # Filter structures
    #df_be = df_be[(df_be['Mean_Eb_all_dft'] > -20.0) & (df_be['Mean_Eb_all_dft'] < -0.2)]
    be_cutoff = 0.1
    logger.info(f"Applying BE cutoff of {be_cutoff} kcal/mol\n")
    df_be = df_be[df_be['Mean_Eb_all_dft'] < be_cutoff]

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
    #logger.info("Welcome to the BEEP Binding Energy Evaluation Platform")
    logger.info(welcome_msg)
    scale_factor = 0.958

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
        df_no_zpve = concatenate_frames(client, mol, ds_w, args.opt_method)

        # Process ZPVE data
        name_hess_be = [f"be_{mol}_{cluster}_{args.opt_method.split('_')[0]}" for cluster in args.hessian_clusters]

        # Retrieve and process ZPVE data for each method
        df_zpve, fit_data_dict  = zpve_correction(name_hess_be, args.be_methods, args.opt_method, client=client, scale_factor=scale_factor)

        # Apply linear models and plot results
        df_zpve_lin = apply_lin_models(df_no_zpve, fit_data_dict, args.be_methods, mol, logger)

        # Calculate and log mean and standard deviation if ZPVE data is available
        res_be_no_zpve, mean, sdev = calculate_mean_std(df_no_zpve, mol, logger)
        res_be_zpve, mean, sdev    = calculate_mean_std(df_zpve, mol, logger)
        res_be_lin_zpve, mean, sdev  = calculate_mean_std(df_zpve_lin, mol, logger)

        #logger.info(f"Binding energy for {mol} is {mean_value:.4f} ± {std_value:.4f}")
        padded_log(logger, "Average binding energy results", padding_char=gear)
        log_dataframe(logger, res_be_no_zpve, f"\nBinding energies without ZPVE correction for {mol}\n")
        log_dataframe(logger, res_be_zpve, f"\nBinding energies with direct ZPVE correction for  {mol}\n")
        log_dataframe(logger, res_be_lin_zpve, f"\nBinding energies with linear model ZPVE correction for {mol}\n")

        en_log_mol = ''
        en_log_mol = write_energy_log(res_be_no_zpve, mol, en_log_mol, "(NO ZPVE):")
        en_log_mol = write_energy_log(res_be_zpve, mol, en_log_mol, "(Direct ZPVE):")
        en_log_mol = write_energy_log(res_be_lin_zpve, mol, en_log_mol, "(Linear model ZPVE):")
        logger.info(en_log_mol)

        final_result_nz = write_energy_log(res_be_no_zpve, mol, final_result_nz, "(NO ZPVE):")
        final_result_dz = write_energy_log(res_be_zpve, mol, final_result_dz, "(Direct ZPVE):")
        final_result_lz = write_energy_log(res_be_lin_zpve, mol, final_result_lz, "(Linear model ZPVE):")
        #en_log_mol = write_energy_log(res_, mol, "")

        padded_log(logger, "Saving all dataframes to csv", padding_char=gear)
        # Save all processed data to a CSV file
        res_be_no_zpve.to_csv(f'{res_folder}/be_no_zpve_{mol}.csv')
        res_be_zpve.to_csv(f'{res_folder}/be_zpve_{mol}.csv')
        res_be_lin_zpve.to_csv(f'{res_folder}/be_lin_zpve_{mol}.csv')

    padded_log(logger, "Summary of binding energy results", padding_char=gear)
    logger.info(final_result_nz)
    logger.info(final_result_dz)
    logger.info(final_result_lz)

if __name__ == "__main__":
    main()

