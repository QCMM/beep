"""
ZPVE vibrational analysis — pure computation using PSI4.

No QCFractal imports. This module contains only the _vibanal_wfn function
which operates on raw Hessian matrices and molecule objects.
"""
import collections
import numpy as np
from typing import Union, Any, Callable
import os, sys
from functools import wraps


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


def _patch_harmonic_analysis():
    """Monkey-patch psi4's harmonic_analysis to skip the MintsHelper/cdsalcs
    block when basisset is None.

    The original code calls ``MintsHelper(basisset)`` to obtain SALCs for
    irrep classification of vibrational modes.  When basisset is None (as in
    BEEP's usage) this segfaults.  The SALCs are only used for labelling
    modes by irrep — frequencies, force constants, and thermodynamics are
    unaffected.  Skipping this block sets ``Uh = {}`` so all modes get
    ``gamma = None`` instead of an irrep label.
    """
    import psi4.driver.qcdb.vib as _vib

    _orig = _vib.harmonic_analysis

    @wraps(_orig)
    def _patched(hess, geom, mass, basisset, irrep_labels, dipder=None,
                 project_trans=True, project_rot=True):
        if basisset is not None:
            return _orig(hess, geom, mass, basisset, irrep_labels,
                         dipder=dipder, project_trans=project_trans,
                         project_rot=project_rot)

        # --- basisset is None: skip MintsHelper, provide empty Uh ---
        import psi4
        _saved = psi4.core.MintsHelper

        class _DummyMintsHelper:
            def __init__(self, *a, **kw):
                pass
            def cdsalcs(self, *a, **kw):
                return self
            def matrix_irrep(self, h):
                return np.zeros((0, 0))

        psi4.core.MintsHelper = _DummyMintsHelper
        try:
            return _orig(hess, geom, mass, None, irrep_labels,
                         dipder=dipder, project_trans=project_trans,
                         project_rot=project_rot)
        finally:
            psi4.core.MintsHelper = _saved

    _vib.harmonic_analysis = _patched


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

    _patch_harmonic_analysis()

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
