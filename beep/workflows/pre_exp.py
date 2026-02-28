"""Pre-exponential factor workflow — refactored from workflows/launch_pre_exp.py."""
import os
import logging

import pandas as pd
from qcportal.client import FractalClient

from ..models.pre_exp import PreExpConfig
from ..core.logging_utils import setup_logging
from ..core.pre_exponential import (
    get_mass, get_sym_num, parse_coordinates,
    align_to_z_axis, get_moments_of_inertia, pre_exponential_factor,
)
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
    qcf.check_collection_existence(client, mol_col)

    main_logger = logging.getLogger("beep")
    main_logger.info(welcome_msg)

    if mol is None:
        mol = list(ds.df.index)

    v_folder = f"./v_{mol_col}"
    os.makedirs(v_folder, exist_ok=True)

    for molecule in mol:
        qcf.check_optimized_molecule(ds, mol_lot, molecule)
        main_logger.info(calculation_msg(mol_col, molecule, mol_lot))

        mol_xyz = qcf.get_xyz(client, mol_col, molecule, mol_lot)
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
