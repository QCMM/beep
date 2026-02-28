"""Pre-exponential factor workflow — refactored from workflows/launch_pre_exp.py."""
import os
import math
import logging

import numpy as np
import pandas as pd
import qcelemental as qcel
from qcportal.client import FractalClient

from ..models.pre_exp import PreExpConfig
from ..core.logging_utils import setup_logging
from ..core.errors import DatasetNotFound, LevelOfTheoryNotFound
from ..adapters import qcfractal_adapter as qcf

welcome_msg = """
---------------------------------------------------------------------------------------
Welcome to the BEEP Range of temperature Pre-Exponential Factor Workflow
---------------------------------------------------------------------------------------

"To deny our impulses is to deny the very thing that makes us human."

                              \u2013 Lana and Lilly Wachowski

---------------------------------------------------------------------------------------

                            By:  Gabriela Silva-Vera  and  Namrata Rani
"""


def calculation_msg(mol_col, mol, level_theory):
    return f"""
---------------------------------
Starting new calculation for {mol}
---------------------------------
Collection: {mol_col}
Molecule: {mol}
Level of theory: {level_theory}
   """


def check_collection_existence(client, collection, collection_type="OptimizationDataset"):
    try:
        qcf.get_collection(client, collection_type, collection)
    except KeyError:
        raise DatasetNotFound(
            f"Collection {collection} does not exist. Please create it first. Exiting..."
        )


def check_optimized_molecule(ds, opt_lot, mol):
    try:
        rr = qcf.fetch_opt_record(ds, mol, opt_lot)
    except KeyError:
        raise LevelOfTheoryNotFound(
            f"{opt_lot} level of theory for {mol} or the entry itself does not exist "
            f"in {ds.name} collection. Add the molecule and optimize it first\n"
        )
    if rr.status == "INCOMPLETE":
        raise ValueError(f" Optimization has status {rr.status} restart it or wait")
    elif rr.status == "ERROR":
        raise ValueError(f" Optimization has status {rr.status} restart it or wait")


def get_xyz(client, dataset, mol_name, level_theory, collection_type="OptimizationDataset"):
    ds_opt = qcf.get_collection(client, collection_type, dataset)
    rr = qcf.fetch_opt_record(ds_opt, mol_name, level_theory)
    mol = rr.get_final_molecule()
    geom = mol.to_string(dtype="xyz")
    xyz_list = geom.splitlines()[2:]
    xyz = '\n'.join(xyz_list)
    return xyz


def get_mass(xyz):
    kg_convert = qcel.constants.get("na") * 1000
    mol = qcel.models.Molecule.from_data(xyz)
    m = mol.masses
    mass_sum = m.sum()
    mass = mass_sum / kg_convert
    return mass


def get_sym_num(xyz):
    import molsym
    schema = qcel.models.Molecule.from_data(xyz).dict()
    mol = molsym.Molecule.from_schema(schema)
    pg, (paxis, saxis) = molsym.find_point_group(mol)
    if pg == 'D0h':
        return pg, 2
    if pg == 'C0v':
        return pg, 1
    else:
        s_m = molsym.Symtext.from_molecule(mol).rotational_symmetry_number
        return pg, s_m


def parse_coordinates(xyz):
    symbols, coordinates = [], []
    for line in xyz.strip().splitlines():
        parts = line.split()
        symbols.append(parts[0])
        coordinates.append(list(map(float, parts[1:])))
    return symbols, np.array(coordinates)


def align_to_z_axis(symbols, coordinates, threshold=1e-8):
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
    kg_convert = qcel.constants.get("na") * 1000
    amu_masses = np.array([qcel.periodictable.to_mass(sym) for sym in symbols])
    masses = amu_masses / kg_convert
    coords = coordinates * qcel.constants.conversion_factor("Angstrom", "m")

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


def pre_exponential_factor(m, T_list, sigma, Ia, Ib, Ic, A):
    kB = qcel.constants.get("kb")
    h = qcel.constants.get("h")
    pi = math.pi

    def _single_T(T):
        translational_part = ((2 * pi * m * kB * T) / h**2) * A
        if Ia == 0:
            rotational_part = (8 * pi**(5/2) * kB * T / h**2) * (Ib / sigma)
        else:
            rotational_part = (pi**0.5 / (sigma * h**3)) * (8 * pi**2 * kB * T)**(3 / 2) * math.sqrt(Ia * Ib * Ic)
        return ((kB * T) / h) * translational_part * rotational_part

    return [_single_T(T) for T in T_list]


def run(config: PreExpConfig, client: FractalClient) -> None:
    mol_col = config.molecule_collection
    mol_lot = config.level_of_theory

    if config.range_of_temperature and len(config.range_of_temperature) == 1:
        T_min = T_max = config.range_of_temperature[0]
        T_list = [T_min]
        T_step = config.temperature_step
    else:
        T_min, T_max = config.range_of_temperature
        T_step = config.temperature_step
        T_list = list(range(T_min, T_max, T_step))

    mol = config.molecule
    A = config.molecule_surface_area

    ds = qcf.get_collection(client, "OptimizationDataset", mol_col)
    check_collection_existence(client, mol_col)

    main_logger = logging.getLogger("beep")
    main_logger.info(welcome_msg)

    if mol is None:
        mol = list(ds.df.index)

    v_folder = f"./v_{mol_col}"
    os.makedirs(v_folder, exist_ok=True)

    for molecule in mol:
        check_optimized_molecule(ds, mol_lot, molecule)
        main_logger.info(calculation_msg(mol_col, molecule, mol_lot))

        mol_xyz = get_xyz(client, mol_col, molecule, mol_lot)
        point_group, sym_num = get_sym_num(mol_xyz)
        symbols, coordinates = parse_coordinates(mol_xyz)
        mol_mass = get_mass(mol_xyz)
        main_logger.info(f"Point group = {point_group}\nSymmetry number = {sym_num}")

        align_coors = align_to_z_axis(symbols, coordinates)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coordinates)
        main_logger.info(
            f"Principal moments of inertia for {molecule} (kg\u00b7m^(2)): "
            f"Ia={Ia:.3e}, Ib={Ib:.3e}, Ic={Ic:.3e}"
        )

        v = pre_exponential_factor(mol_mass, T_list, sym_num, Ia, Ib, Ic, A)
        main_logger.info(
            f"Pre-exponential factor for {molecule} in the range of "
            f"{T_min}K to {T_max}K with steps of {T_step}K has been calculated"
        )

        v_log_file = f"v_{mol_col}/v_{molecule}.dat"
        v_logger = logging.getLogger(f"v_{molecule}")
        v_logger.setLevel(logging.INFO)

        v_handler = logging.FileHandler(v_log_file)
        v_handler.setFormatter(logging.Formatter(" %(message)s"))
        v_logger.addHandler(v_handler)

        table = pd.DataFrame({"T": T_list, "v": v})
        v_logger.info("\n" + table.to_string(index=False))
