"""Pre-exponential factor workflow — refactored from workflows/launch_pre_exp.py."""
import json
import logging
from pathlib import Path

import pandas as pd
from qcportal.client import FractalClient

from ..models.pre_exp import PreExpConfig
from ..core.pre_exponential import (
    get_mass, get_sym_num, parse_coordinates,
    align_to_z_axis, get_moments_of_inertia, pre_exponential_factor,
)
from ..core.logging_utils import beep_banner
from ..adapters import qcfractal_adapter as qcf

welcome_msg = beep_banner(
    "Pre-Exponential Factor",
    quote="To deny our impulses is to deny the very thing that makes us human.",
    quote_author="Lana and Lilly Wachowski",
    authors="Gabriela Silva-Vera and Namrata Rani",
)


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

    if mol is None:
        mol = list(ds.df.index)

    # Use first molecule name for the output folder (or collection name if list)
    folder_label = mol[0] if isinstance(mol, list) else mol
    res_folder = Path.cwd() / folder_label / "pre_exp"
    res_folder.mkdir(parents=True, exist_ok=True)

    # File logging inside the output folder
    log_file = res_folder / f"beep_pre_exp_{folder_label}.log"
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    main_logger.addHandler(file_handler)

    # Save a copy of the input config
    config_path = res_folder / f"pre_exp_{folder_label}.json"
    config_path.write_text(json.dumps(config.dict(), indent=4, default=str))

    main_logger.info(welcome_msg)

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

        table = pd.DataFrame({"T": T_list, "v": v})

        # Write per-molecule data file into the output folder
        v_dat_file = res_folder / f"v_{molecule}.dat"
        v_dat_file.write_text(table.to_string(index=False))
        main_logger.info(f"Saved pre-exponential data to {v_dat_file}")
        main_logger.info("\n" + table.to_string(index=False))

    main_logger.removeHandler(file_handler)
    file_handler.close()
