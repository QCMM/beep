import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Union, Any, List, Tuple, Callable
import pandas as pd
import os,sys
from functools import wraps

import qcfractal.interface as ptl
from qcfractal.interface.client import FractalClient
from qcelemental.models.molecule import Molecule
from qcfractal.interface.collections import Dataset, OptimizationDataset, ReactionDataset
from .utils.logging_utils import *

def suppress_stdout(func: Callable) -> Callable:
    """
    Decorator to suppress stdout output of a function. Any output to the console 
    during the execution of the wrapped function will be discarded.

    Args:
        func (Callable): The function whose stdout output should be suppressed.

    Returns:
        Callable: The wrapped function with suppressed stdout output.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
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

def get_zpve_mol(client: ptl.FractalClient, mol: int, lot_opt: str, on_imaginary: str = 'return') -> Union[bool, Any]:
    """
    Computes vibrational analysis for a given molecule and handles imaginary frequencies.

    Parameters:
    ----------
    client : ptl.FractalClient
        The Fractal client for interacting with the QCArchive.
    mol : int
        The molecule ID for which vibrational analysis is performed.
    lot_opt : str
        The level of theory and basis set, e.g., 'HF_6-31G'.
    on_imaginary : str, optional
        Specifies behavior when imaginary frequencies are found.
        - 'return': Return False.
        - 'raise': Raise a ValueError.
        Defaults to 'return'.

    Returns:
    -------
    Union[bool, Any]
        The computed zero-point vibrational energy (ZPVE) data if no imaginary frequencies are found or `False` if 
        imaginary frequencies are found and `on_imaginary='return'`. Returns a tuple containing the ZPVE and a boolean
        indicating success (`True` if no imaginary frequencies, `False` otherwise).

    Raises:
    ------
    ValueError
        If `on_imaginary='raise'` and imaginary frequencies are found.
    """
    logger = logging.getLogger("beep")

    mol_obj = client.query_molecules(mol)[0]
    num_atm = len(mol_obj.symbols)

    if num_atm == 1:
        logger.info(f"Molecule {mol} is an atom, will retrun 0.0 for ZPVE")
        return 0.0, True

    mol_form = mol_obj.dict()['identifiers']['molecular_formula']
    method = lot_opt.split("_")[0]
    basis = lot_opt.split("_")[1]
    if method[0] == 'U':
        method = method[1:]

    try:
        result = client.query_results(
            driver='hessian',
            molecule=mol,
            method=method,
            basis=basis
        )[0]
        logger.info(f"Molecule {mol} with molecular formula {mol_form} has a Hessian calculation.")
    except IndexError:
        logger.info(f"Molecule {mol} with molecular formula {mol_form} has not finished computing.")
        return None, True

    hess = result.dict()['return_result']
    energy = result.dict()['extras']['qcvars']['CURRENT ENERGY']
    vib, therm = _vibanal_wfn(hess=hess, molecule=result.get_molecule(), energy=energy)

    # Check for imaginary frequencies
    imag_freqs = [num for num in vib['omega'].data if abs(num.imag) > 1]
    if imag_freqs:
        # Dump Hessian to a text file if there are imaginary frequencies
        if on_imaginary == 'raise':
            np.savetxt(f'{mol}_hess.dat', hess, fmt='%.18e')
            raise ValueError(f"There are imaginary frequencies: {imag_freqs}. You need to reoptimize {mol}.")
        elif on_imaginary == 'return':
            logger.info(f"There are imaginary frequencies: {imag_freqs}, proceed with caution.")
            return therm['ZPE_vib'].data, False
        else:
            raise ValueError(f"Invalid option for on_imaginary: {on_imaginary}")

    return therm['ZPE_vib'].data, True


