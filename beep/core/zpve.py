"""
ZPVE vibrational analysis — pure numpy implementation.

No psi4 or QCFractal imports. Computes vibrational frequencies and
thermochemistry from a Hessian matrix and a qcelemental Molecule.
"""
import numpy as np
from typing import Union
from types import SimpleNamespace
import qcelemental as qcel
from qcelemental import Datum


# ---------------------------------------------------------------------------
# Physical constants (CODATA 2014 via qcelemental)
# ---------------------------------------------------------------------------
_NA = qcel.constants.na              # Avogadro [mol^-1]
_H = qcel.constants.h                # Planck [J·s]
_C = qcel.constants.c                # speed of light [m/s]
_KB = qcel.constants.kb              # Boltzmann [J/K]
_R = qcel.constants.R                # gas constant [J/(mol·K)]
_EH = qcel.constants.hartree2J       # Hartree [J]
_A0 = qcel.constants.bohr2angstroms  # Bohr radius [Å]
_A0_M = _A0 * 1e-10                  # Bohr radius [m]
_AMU2KG = qcel.constants.amu2kg      # atomic mass unit [kg]
_HARTREE2KJMOL = qcel.constants.hartree2kJmol

# Conversion: sqrt(eigenvalue in Eh/(a0^2·amu)) → cm^-1
_UCONV_CM1 = np.sqrt(_EH / (_A0_M**2 * _AMU2KG)) / (2.0 * np.pi * _C * 100.0)

# Conversion: R in Eh/K (for thermo sums in reduced temperature units)
_R_EH_K = _R / (_HARTREE2KJMOL * 1000.0)  # [Eh/(mol·K)] / mol → Eh/K


# ---------------------------------------------------------------------------
# TR space construction
# ---------------------------------------------------------------------------

def _build_tr_space(geom, mass, project_trans=True, project_rot=True):
    """Build orthonormal translation/rotation subspace in mass-weighted coords.

    Parameters
    ----------
    geom : (nat, 3) ndarray — Cartesian geometry in bohr
    mass : (nat,) ndarray — atomic masses in amu
    project_trans : bool — include translation vectors
    project_rot : bool — include rotation vectors

    Returns
    -------
    TRorth : (nrt, 3*nat) ndarray — orthonormal TR basis vectors
    """
    nat = len(mass)
    vectors = []

    if project_trans:
        for i in range(3):
            t = np.zeros(3 * nat)
            for a in range(nat):
                t[3 * a + i] = np.sqrt(mass[a])
            vectors.append(t)

    if project_rot:
        com = np.average(geom, weights=mass, axis=0)
        r = geom - com
        for a in range(nat):
            sm = np.sqrt(mass[a])
            # Rx: L_x = y*pz - z*py
            rx = np.zeros(3 * nat)
            rx[3 * a + 1] = sm * r[a, 2]
            rx[3 * a + 2] = -sm * r[a, 1]
            # Ry: L_y = z*px - x*pz
            ry = np.zeros(3 * nat)
            ry[3 * a + 0] = -sm * r[a, 2]
            ry[3 * a + 2] = sm * r[a, 0]
            # Rz: L_z = x*py - y*px
            rz = np.zeros(3 * nat)
            rz[3 * a + 0] = sm * r[a, 1]
            rz[3 * a + 1] = -sm * r[a, 0]
        # Accumulate one vector per rotation axis (sum over atoms)
        Rx = np.zeros(3 * nat)
        Ry = np.zeros(3 * nat)
        Rz = np.zeros(3 * nat)
        for a in range(nat):
            sm = np.sqrt(mass[a])
            Rx[3 * a + 1] += sm * r[a, 2]
            Rx[3 * a + 2] += -sm * r[a, 1]
            Ry[3 * a + 0] += -sm * r[a, 2]
            Ry[3 * a + 2] += sm * r[a, 0]
            Rz[3 * a + 0] += sm * r[a, 1]
            Rz[3 * a + 1] += -sm * r[a, 0]
        vectors.extend([Rx, Ry, Rz])

    if not vectors:
        return np.zeros((0, 3 * nat))

    TRspace = np.vstack([v[np.newaxis, :] for v in vectors])
    # Orthogonalize via SVD, discard linearly dependent vectors
    _, s, Vt = np.linalg.svd(TRspace, full_matrices=False)
    tol = 1e-6 * s[0] if len(s) > 0 else 0.0
    nrt = int(np.sum(s > tol))
    return Vt[:nrt]


# ---------------------------------------------------------------------------
# Harmonic analysis
# ---------------------------------------------------------------------------

def _harmonic_analysis(hess, geom, mass, project_trans=True, project_rot=True):
    """Mass-weight, project TR, diagonalize → frequencies and mode info.

    Parameters
    ----------
    hess : (3N, 3N) ndarray — non-mass-weighted Hessian [Eh/a0^2]
    geom : (nat, 3) ndarray — geometry [bohr]
    mass : (nat,) ndarray — masses [amu]

    Returns
    -------
    vibinfo : dict of Datum — vibrational information
    """
    nat = len(mass)
    ndof = 3 * nat

    # Mass-weight the Hessian
    sqrtm = np.repeat(np.sqrt(mass), 3)
    sqrtminv = 1.0 / sqrtm
    mwhess = np.einsum('i,ij,j->ij', sqrtminv, hess, sqrtminv)

    # Build TR projector
    TRorth = _build_tr_space(geom, mass, project_trans, project_rot)
    P = np.eye(ndof)
    for v in TRorth:
        P -= np.outer(v, v)

    # Project and diagonalize
    mwhess_proj = P @ mwhess @ P
    eigvals, eigvecs = np.linalg.eigh(mwhess_proj)

    # Convert eigenvalues to frequencies [cm^-1]
    # scimath.sqrt gives complex values for negative eigenvalues (imaginary freqs)
    omega = np.lib.scimath.sqrt(eigvals) * _UCONV_CM1

    # Classify modes: TR modes are near-zero
    trv = []
    nrt = len(TRorth)
    # Sort eigenvalues by magnitude to identify the nrt smallest as TR
    mag_order = np.argsort(np.abs(eigvals))
    is_tr = set(mag_order[:nrt].tolist())
    for i in range(ndof):
        if i in is_tr:
            trv.append("TR")
        else:
            trv.append("V")

    # Reduced masses
    wL = np.einsum('i,ij->ij', sqrtminv, eigvecs)  # un-mass-weighted modes
    norms_sq = np.sum(wL**2, axis=0)
    norms_sq = np.where(norms_sq > 1e-30, norms_sq, 1e-30)
    mu = 1.0 / norms_sq  # reduced mass [amu]

    # Characteristic vibrational temperature
    theta_vib = np.abs(omega.real) * 100.0 * _C * _H / _KB  # [K]

    vibinfo = {
        "omega": Datum("omega", "cm^-1", omega),
        "mu": Datum("mu", "u", mu),
        "TRV": SimpleNamespace(data=trv),
        "theta_vib": Datum("theta_vib", "K", theta_vib),
    }
    return vibinfo


# ---------------------------------------------------------------------------
# Rotational properties
# ---------------------------------------------------------------------------

def _rotational_properties(geom, mass):
    """Compute rotational constants and rotor type from geometry and masses.

    Parameters
    ----------
    geom : (nat, 3) ndarray — geometry [bohr]
    mass : (nat,) ndarray — masses [amu]

    Returns
    -------
    rot_const : (3,) ndarray — rotational constants [cm^-1]
    rotor_type : str — "RT_ATOM", "RT_LINEAR", or "RT_ASYMMETRIC_TOP"
    """
    nat = len(mass)
    if nat == 1:
        return np.zeros(3), "RT_ATOM"

    # Center of mass
    com = np.average(geom, weights=mass, axis=0)
    r = geom - com

    # Inertia tensor in amu·bohr^2
    I = np.zeros((3, 3))
    for a in range(nat):
        rsq = np.dot(r[a], r[a])
        I += mass[a] * (rsq * np.eye(3) - np.outer(r[a], r[a]))

    # Principal moments
    eigvals = np.linalg.eigvalsh(I)
    eigvals = np.sort(eigvals)  # Ia <= Ib <= Ic

    # Convert to SI: I_SI = I_amu_bohr2 * amu2kg * a0_m^2
    I_SI = eigvals * _AMU2KG * _A0_M**2

    # Rotational constants: B = h / (8*pi^2*c*I) in cm^-1
    rot_const = np.zeros(3)
    for i in range(3):
        if I_SI[i] > 1e-50:
            rot_const[i] = _H / (8.0 * np.pi**2 * _C * 100.0 * I_SI[i])

    # Determine rotor type
    n_nonzero = np.sum(eigvals > 1e-10)
    if n_nonzero == 0:
        rotor_type = "RT_ATOM"
    elif n_nonzero <= 2 and eigvals[0] < 1e-10:
        rotor_type = "RT_LINEAR"
    else:
        rotor_type = "RT_ASYMMETRIC_TOP"

    return rot_const, rotor_type


# ---------------------------------------------------------------------------
# Thermochemistry
# ---------------------------------------------------------------------------

def _thermo(vibinfo, T, P, multiplicity, molecular_mass, E0, sigma,
            rot_const, rotor_type):
    """Standard statistical mechanics thermochemistry.

    Parameters
    ----------
    vibinfo : dict — from _harmonic_analysis
    T : float — temperature [K]
    P : float — pressure [Pa]
    multiplicity : int — spin multiplicity
    molecular_mass : float — total mass [amu]
    E0 : float — electronic energy [Eh]
    sigma : int — rotational symmetry number
    rot_const : (3,) ndarray — rotational constants [cm^-1]
    rotor_type : str

    Returns
    -------
    therminfo : dict of Datum
    """
    # --- Electronic ---
    S_elec = np.log(multiplicity)  # in units of R
    Cv_elec = 0.0
    Cp_elec = 0.0
    ZPE_elec = 0.0
    E_elec = 0.0
    H_elec = 0.0

    # --- Translational (Sackur-Tetrode) ---
    beta = 1.0 / (_KB * T)
    M_kg = molecular_mass * _AMU2KG
    q_trans = ((2.0 * np.pi * M_kg / (beta * _H**2))**1.5) * _KB * T / P

    S_trans = 2.5 + np.log(q_trans)
    Cv_trans = 1.5
    Cp_trans = 2.5
    ZPE_trans = 0.0
    E_trans = 1.5 * T   # in K (multiply by R to get energy)
    H_trans = 2.5 * T

    # --- Rotational (rigid rotor) ---
    if rotor_type == "RT_ATOM":
        S_rot = 0.0
        Cv_rot = 0.0
        Cp_rot = 0.0
        E_rot = 0.0
    elif rotor_type == "RT_LINEAR":
        # Linear molecule: B is rot_const[1] (the middle value)
        B = rot_const[1] if rot_const[1] > 0 else rot_const[2]
        q_rot = 1.0 / (sigma * beta * 100.0 * _C * _H * B) if B > 0 else 1.0
        S_rot = 1.0 + np.log(q_rot)
        Cv_rot = 1.0
        Cp_rot = 1.0
        E_rot = T
    else:
        # Nonlinear molecule
        phi = rot_const * 100.0 * _C * _H / _KB  # char. rotational temps [K]
        phi_prod = np.prod(phi[phi > 0])
        if phi_prod > 0:
            q_rot = np.sqrt(np.pi) * T**1.5 / (sigma * np.sqrt(phi_prod))
        else:
            q_rot = 1.0
        S_rot = 1.5 + np.log(q_rot)
        Cv_rot = 1.5
        Cp_rot = 1.5
        E_rot = 1.5 * T

    ZPE_rot = 0.0
    H_rot = E_rot

    # --- Vibrational (quantum harmonic oscillator) ---
    trv = vibinfo["TRV"].data
    omega = vibinfo["omega"].data

    # Filter to real vibrational modes only
    vib_theta = []
    for i, mode_type in enumerate(trv):
        if mode_type != "V":
            continue
        w = omega[i]
        # Skip imaginary modes (imag > real)
        if abs(w.imag) > abs(w.real):
            continue
        if abs(w.real) < 1.0:  # skip near-zero modes
            continue
        theta = abs(w.real) * 100.0 * _C * _H / _KB
        vib_theta.append(theta)

    vib_theta = np.array(vib_theta) if vib_theta else np.array([])

    if len(vib_theta) > 0:
        rT = vib_theta / T  # reduced temperatures
        # Protect against overflow
        rT_safe = np.minimum(rT, 500.0)

        ZPE_vib_K = np.sum(vib_theta) / 2.0  # in K
        E_vib_K = ZPE_vib_K + np.sum(vib_theta / np.expm1(rT_safe))
        S_vib = np.sum(rT_safe / np.expm1(rT_safe) - np.log(1.0 - np.exp(-rT_safe)))
        Cv_vib = np.sum(np.exp(rT_safe) * (rT_safe / np.expm1(rT_safe))**2)
    else:
        ZPE_vib_K = 0.0
        E_vib_K = 0.0
        S_vib = 0.0
        Cv_vib = 0.0

    Cp_vib = Cv_vib
    H_vib_K = E_vib_K

    # --- Convert to Eh ---
    # S, Cv, Cp: dimensionless (in units of R) → multiply by R_Eh_K for mEh/K
    # Energy terms in K → multiply by R_Eh_K for Eh

    # ZPE
    ZPE_vib = ZPE_vib_K * _R_EH_K
    ZPE_corr = ZPE_elec + ZPE_trans + ZPE_rot + ZPE_vib
    ZPE_tot = E0 + ZPE_corr

    # Thermal energy
    E_vib = E_vib_K * _R_EH_K
    E_corr = (E_elec + E_trans + E_rot) * _R_EH_K + E_vib
    E_tot = E0 + E_corr

    # Enthalpy
    H_vib = H_vib_K * _R_EH_K
    H_corr = (H_elec + H_trans + H_rot) * _R_EH_K + H_vib
    H_tot = E0 + H_corr

    # Gibbs: G = H - TS
    S_tot = S_elec + S_trans + S_rot + S_vib  # in units of R
    G_corr = H_corr - T * S_tot * _R_EH_K
    G_tot = E0 + G_corr

    therminfo = {
        "ZPE_vib":  Datum("ZPE_vib",  "Eh", ZPE_vib),
        "ZPE_corr": Datum("ZPE_corr", "Eh", ZPE_corr),
        "ZPE_tot":  Datum("ZPE_tot",  "Eh", ZPE_tot),
        "E_corr":   Datum("E_corr",   "Eh", E_corr),
        "E_tot":    Datum("E_tot",    "Eh", E_tot),
        "H_corr":   Datum("H_corr",   "Eh", H_corr),
        "H_tot":    Datum("H_tot",    "Eh", H_tot),
        "G_corr":   Datum("G_corr",   "Eh", G_corr),
        "G_tot":    Datum("G_tot",    "Eh", G_tot),
    }
    return therminfo


# ---------------------------------------------------------------------------
# Public interface (same signature as before)
# ---------------------------------------------------------------------------

def _vibanal_wfn(hess=None, irrep=None, molecule=None, energy=None,
                 project_trans=True, project_rot=True, **kwargs):
    """Vibrational analysis and thermochemistry from a Hessian matrix.

    Parameters
    ----------
    hess : (3N, 3N) ndarray
        Non-mass-weighted Hessian in atomic units [Eh/a0^2].
    molecule : qcelemental.models.Molecule
        Molecule with geometry (bohr) and masses (amu).
    energy : float
        Electronic energy at the minimum [Eh].
    irrep : int or str, optional
        If given, thermochemistry is skipped.
    project_trans : bool
        Project out translations (default True).
    project_rot : bool
        Project out rotations (default True).

    Returns
    -------
    vibinfo : dict of Datum
        Vibrational frequencies and mode data.
    therminfo : dict of Datum
        Thermochemical corrections and totals.
    """
    geom = np.asarray(molecule.geometry)   # (nat, 3) in bohr
    mass = np.asarray(molecule.masses)     # (nat,) in amu

    vibinfo = _harmonic_analysis(
        np.asarray(hess), geom, mass,
        project_trans=project_trans, project_rot=project_rot,
    )

    therminfo = None
    if irrep is None:
        rot_const, rotor_type = _rotational_properties(geom, mass)
        therminfo = _thermo(
            vibinfo,
            T=298.15,
            P=101325.0,
            multiplicity=molecule.molecular_multiplicity,
            molecular_mass=np.sum(mass),
            E0=energy,
            sigma=1,  # C1 symmetry default (appropriate for asymmetric clusters)
            rot_const=rot_const,
            rotor_type=rotor_type,
        )

    return vibinfo, therminfo
