"""
Pre-exponential factor — pure computation functions.

Computes rotational/translational partition functions and pre-exponential
desorption rate factors. No QCFractal server dependencies.
"""
import math

import numpy as np
import qcelemental as qcel


def get_mass(xyz):
    """Get total molecular mass in kg from XYZ string."""
    kg_convert = qcel.constants.get("na") * 1000
    mol = qcel.models.Molecule.from_data(xyz)
    m = mol.masses
    mass_sum = m.sum()
    mass = mass_sum / kg_convert
    return mass


def get_sym_num(xyz):
    """Get point group and rotational symmetry number from XYZ string."""
    import molsym
    schema = qcel.models.Molecule.from_data(xyz).dict()
    mol = molsym.Molecule.from_schema(schema)
    pg, (paxis, saxis) = molsym.find_point_group(mol)
    if pg == 'D0h':
        return pg, 2
    if pg == 'C0v':
        return pg, 1
    else:
        mol.tol = 1e-4  # Relax tolerance for optimized geometries
        s_m = molsym.Symtext.from_molecule(mol).rotational_symmetry_number
        return pg, s_m


def parse_coordinates(xyz):
    """Parse XYZ string into symbols list and coordinates array."""
    symbols, coordinates = [], []
    for line in xyz.strip().splitlines():
        parts = line.split()
        symbols.append(parts[0])
        coordinates.append(list(map(float, parts[1:])))
    return symbols, np.array(coordinates)


def align_to_z_axis(symbols, coordinates, threshold=1e-8):
    """Align molecule to principal axes via SVD, zeroing near-zero components."""
    masses = np.array([qcel.periodictable.to_mass(sym) for sym in symbols])
    total_mass = np.sum(masses)
    center_of_mass = np.sum(masses[:, np.newaxis] * coordinates, axis=0) / total_mass
    shifted_coords = coordinates - center_of_mass
    _, _, vh = np.linalg.svd(shifted_coords)
    rotation_matrix = vh.T
    aligned_coords = np.dot(shifted_coords, rotation_matrix)
    aligned_coords[np.abs(aligned_coords) < threshold] = 0.0
    return aligned_coords


def get_moments_of_inertia(symbols, coordinates):
    """Compute principal moments of inertia (kg*m^2) from symbols and coordinates (Angstrom).

    The inertia tensor is taken about the molecule's centre of mass: a
    COM shift is applied unconditionally before evaluating the tensor.
    This makes the function correct regardless of whether the caller
    already aligned/centred the geometry — without the shift, any COM
    offset in the input geometry inflates every eigenvalue via the
    parallel-axis theorem (e.g. ~18% drift on a typical H2O optimised
    geometry).
    """
    kg_convert = qcel.constants.get("na") * 1000
    amu_masses = np.array([qcel.periodictable.to_mass(sym) for sym in symbols])
    masses = amu_masses / kg_convert
    coords = coordinates * qcel.constants.conversion_factor("Angstrom", "m")

    com = (masses[:, None] * coords).sum(0) / masses.sum()
    coords = coords - com

    I = np.zeros((3, 3))
    for m, r in zip(masses, coords):
        I[0, 0] += m * (r[1]**2 + r[2]**2)
        I[1, 1] += m * (r[0]**2 + r[2]**2)
        I[2, 2] += m * (r[0]**2 + r[1]**2)
        I[0, 1] -= m * r[0] * r[1]
        I[0, 2] -= m * r[0] * r[2]
        I[1, 2] -= m * r[1] * r[2]

    I[1, 0], I[2, 0], I[2, 1] = I[0, 1], I[0, 2], I[1, 2]
    eigenvalues, _ = np.linalg.eigh(I)
    Ia, Ib, Ic = np.sort(eigenvalues)
    return Ia, Ib, Ic


# Relative threshold below which the smallest principal moment of inertia is
# treated as zero, i.e. the molecule is linear. Diagonalising the inertia
# tensor of a linear molecule gives Ia ~ 1e-60 kg m^2 (round-off, possibly
# negative) rather than exactly 0, so an exact comparison is not reliable.
LINEAR_INERTIA_REL_TOL = 1e-6


def is_linear_rotor(Ia, Ib, rel_tol=LINEAR_INERTIA_REL_TOL):
    """True when ``Ia`` is negligible relative to ``Ib`` (linear molecule)."""
    return Ia < rel_tol * Ib


def pre_exponential_factor(m, T_list, sigma, Ia, Ib, Ic, A):
    """Compute pre-exponential desorption rate factor for a list of temperatures.

    Transition-state-theory prefactor (Tait et al. 2005; Minissale et al.
    2022, ACS Earth Space Chem. 6, 597, Eqs. 16-20)::

        nu = (kB T / h) * q_trans,2D * q_rot

    with ``q_trans,2D = A * 2 pi m kB T / h^2`` and the classical rigid-rotor
    partition functions. A linear rotor (``Ia`` negligible relative to
    ``Ib``, see :func:`is_linear_rotor`) uses::

        q_rot,lin = 8 pi^2 I kB T / (sigma h^2)

    a nonlinear one::

        q_rot,3D = sqrt(pi) / (sigma h^3) * (8 pi^2 kB T)^(3/2) * sqrt(Ia Ib Ic)

    Note: Minissale et al. 2022 Eq. 20 carries an extra ``sqrt(pi)`` in the
    linear form and their Table 4 an extra ``pi``; both are leftovers of the
    three-dimensional orientation volume (8 pi^2) that has no counterpart for
    the two angles of a linear molecule (orientation volume 4 pi). The form
    used here equals the high-temperature limit ``kB T / (sigma h c B)`` of
    the quantum rotor (CO at 298 K: 107.3 for B = 1.9313 cm^-1); the Eq. 20
    form would give 190 and the Table 4 form 337.

    ``A`` is the surface area per adsorbed molecule in m^2 (1e-19 m^2 for
    most small molecules, Minissale et al. 2022).
    """
    kB = qcel.constants.get("kb")
    h = qcel.constants.get("h")
    pi = math.pi
    linear = is_linear_rotor(Ia, Ib)
    if not linear and (Ia < 0 or Ib < 0 or Ic < 0):
        raise ValueError(
            f"Principal moments of inertia must be non-negative; got Ia={Ia}, Ib={Ib}, Ic={Ic}"
        )

    def _single_T(T):
        translational_part = ((2 * pi * m * kB * T) / h**2) * A
        if linear:
            rotational_part = (8 * pi**2 * kB * T / h**2) * (Ib / sigma)
        else:
            rotational_part = (pi**0.5 / (sigma * h**3)) * (8 * pi**2 * kB * T)**(3 / 2) * math.sqrt(Ia * Ib * Ic)
        return ((kB * T) / h) * translational_part * rotational_part

    return [_single_T(T) for T in T_list]
