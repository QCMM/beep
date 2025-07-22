import sys, time, argparse, logging
import math
import numpy as np       # needed by molsym
import qcfractal.interface as ptl
import qcelemental as qcel # needed by molsym
import molsym # for the symmetry point
from beep.utils import logging_utils as bp_log

welcome_msg = """       
                  Welcome to the Pre-Exponential factor calculator! 

Description: The pre-ext uses the xxxxxx formula uses xxxxxxx

Author: b-unnit, namrata-rani10
            """


def calculation_msg(
    mol_col: str, mol: str, level_theory: str
) -> str:
    """
    Format a message for the start of the pre-exponential factor calculation.

    Args:
    - mol_col: Name of the molecule collection.
    - mol: Name of the target molecule.
    - level_theory: Method and basis at wich the molecule is optimized.

    Returns:
    - Formatted message string.
    """
    return f"""
-----------------------------------------------------------------------------------------
Starting the calculation of the pre-exponential factor of the molecule {mol} in the collection {mol_col} optimized at the {level_theory} level of theory
-----------------------------------------------------------------------------------------
    """


def parse_arguments(
) -> argparse.Namespace:
    """
    Parse command line arguments for the script.

    Returns:
    - Namespace containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="""
A command line interface to calculate the pre-exponential factor of a given molecule inside of a collcetion.
    """
    )

    parser.add_argument(
        "--client_address",
        default="localhost:7777",
        help="The URL address and port of the QCFractal server (default: localhost:7777)",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="The username for the database client (Default = None)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="The password for the database client (Default = None)",
    )
    parser.add_argument(
        "--molecule",
        nargs='+',
        default=None,
        help="Molecule to be sampled (from a QCFractal OptimizationDataSet collection). None calculates all molecules in a collection (default: None)",
    )
    parser.add_argument(
        "--molecule_collection",
        default="Small_molecules",
        help="The name of the collection containing molecules or radicals (default: Small_molecules)",
    )
    parser.add_argument(
        "--level_of_theory",
        default="blyp_def2-svp",
        help="The level of theory in which the molecule is optimized, in the format: method_basis (default: blyp_def2-svp)",
    )          
    parser.add_argument(
        "--range_of_temperature",
        default="10 273",
        help="Range of temperature in K (default: 10 273)",
    )  

  
    return parser.parse_args()


def check_collection_existence(
    client: FractalClient,
    collection: List,
    collection_type: str = "OptimizationDataset",
) -> None:
    """
    Check the existence of collections and raise DatasetNotFound error if not found.

    Args:
    - client: QCFractal client object
    - collection: QCFractal Datasets.
    - collection_type: type of Optimization Dataset

    Raises:
    - DatasetNotFound: If any of the specified collections do not exist.
    """
    try:
        client.get_collection(collection_type, collection)
    except KeyError:
        raise DatasetNotFound(
            f"Collection {collection} does not exist. Please create it first. Exiting..."
        )  


def check_optimized_molecule(
    ds: OptimizationDataset, opt_lot: str, mol: str
) -> None:
    """
    Check if all molecules are optimized at the requested level of theory.

    Args:
    - ds: OptimizationDataset containing the optimization records.
    - opt_lot: Level of theory string.
    - mol: Molecule name to check.

    Raises:
    - LevelOfTheoryNotFound: If the level of theory for a molecule or the entry itself does not exist.
    - ValueError: If optimizations are incomplete or encountered an error.
    """
    try:
        rr = ds.get_record(mol, opt_lot)
    except KeyError:
        raise LevelOfTheoryNotFound(
            f"{opt_lot} level of theory for {mol} or the entry itself does not exist in {ds.name} collection. Add the molecule and optimize it first\n"
        )
    if rr.status == "INCOMPLETE":
        raise ValueError(f" Optimization has status {rr.status} restart it or wait")
    elif rr.status == "ERROR":
        raise ValueError(f" Optimization has status {rr.status} restart it or wait")


def get_xyz(
    dataset: str, mol_name: str, level_theory: str, collection_type: str = "OptimizationDataset"
) -> str:
    """
    Extract the xyz of the molecule

    Args:
    - dataset: dataset containing the molecule.
    - mol_name: molecule name in the dataset.
    - level_theory: Level of theory at which the molecule is optimized.
    - collection_type: Type of optimization dataset (Default = OptimizationDataset.

    Returns:
    - XYZ file, excluding total number of atoms, charge, multiplicity and number of atom for each element present
    """
    #client = ptl.FractalClient(address=client_address, username = username, password = password, verify=False)
    ds_opt = client.get_collection(collection_type,dataset)
    rr = ds_opt.get_record(mol_name, level_theory)
    mol = rr.get_final_molecule()    
    geom = mol.to_string(dtype="xyz")
    xyz_list = geom.splitlines()[2:]     
    xyz = '\n'.join(xyz_list)
    return(xyz)


def get_mass(
    xyz: str
) -> float:
    """
    Calculates the mass of a molecule using the xyz.
    """
    mol = qcel.models.Molecule.from_data(xyz)
    m = mol.masses
    mass = m.sum()
    return(mass)


def sym_num(
    xyz: str
) -> int:
    """
    Gives the symmetry number from the xyz (doesnt work with linear molecules)
    Symmetry numbers given by tables By P. W. ATKINS, M. S. CHILD, and C. S. G. PHILLIPS
    
    Args:
    - xyz: xyz file (only coordinates, no multiplicity, charge, etc)

    Returns:
    - Symmetry number
    """
    
    group_to_number = {
        "C1": 1, "Cs": 1, "Ch": 1, "Ci": 1, "S2": 1,
        "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "C8": 8,
        "D2": 4, "D3": 6, "D4": 8, "D5": 10, "D6": 12,
        "C2v": 2, "C3v": 3, "C4v": 4, "C5v": 5, "C6v": 6,
        "C2h": 2, "C3h": 3, "C4h": 4, "C5h": 5, "C6h": 6,
        "D2h": 4, "D3h": 6, "D4h": 8, "D5h": 10, "D6h": 12,
        "D2d": 4, "D3d": 6, "D4d": 8, "D5d": 10, "D6d": 12,
        "S4": 2, "S6": 3, "S8": 4,
        "T": 12, "Td": 12, "Th": 12, "O": 24, "Oh": 24,
        "I": 60, "Ih": 60,
    }

    schema = qcel.models.Molecule.from_data(xyz).dict()
    mol = molsym.Molecule.from_schema(schema)
    pg, (paxis, saxis) = molsym.find_point_group(mol)

    if pg not in group_to_number:
        raise Keyerror(f"Group {pg} not found in the dictionary... that shouldn't happen")

    return(group_to_number.get(pg))


def parse_coordinates(
  xyz: str
) -> list:
    """
    Parse atomic symbols and their xyz coordinates.
    """
  
    symbols, coordinates = [], []
    for line in input_string.strip().splitlines():
        parts = line.split()
        symbols.append(parts[0])
        coordinates.append(list(map(float, parts[1:])))
    return symbols, np.array(coordinates)


def align_to_z_axis(
  symbols: list, coordinates: list, threshold=1e-8
):
    """
    Align the molecule along the z-axis and zero out small values across all axes.
    """
  
    masses = np.array([qcel.periodictable.to_mass(sym) for sym in symbols]) 
    total_mass = np.sum(masses)

    # Center the molecule at the origin (set center of mass to [0, 0, 0])
    center_of_mass = np.sum(masses[:, np.newaxis] * coordinates, axis=0) / total_mass
    shifted_coords = coordinates - center_of_mass

    # Use SVD to find the best alignment axis (SVD is numpy function to get the rotatational matrix to rotate the coordinates along z-axis)
    _, _, vh = np.linalg.svd(shifted_coords)
    rotation_matrix = vh.T

    # Rotate the coordinates to align the molecule along the z-axis
    aligned_coords = np.dot(shifted_coords, rotation_matrix)

    # Zero out small values based on the threshold for all axes
    aligned_coords[np.abs(aligned_coords) < threshold] = 0.0

    return aligned_coords


def get_moments_of_inertia(
  symbols: list, coordinates: list
) -> float:
    """
    Calculate the moments of inertia after alignment.
    """
  
    masses = np.array([qcel.periodictable.to_mass(sym) for sym in symbols])
    coords = coordinates * qcel.constants.conversion_factor("Angstrom", "m")  # Convert to meters

    # Initialize the inertia tensor
    I = np.zeros((3, 3))
    for m, r in zip(masses, coords):
        I[0, 0] += m * (r[1]**2 + r[2]**2)
        I[1, 1] += m * (r[0]**2 + r[2]**2)
        I[2, 2] += m * (r[0]**2 + r[1]**2)
        I[0, 1] -= m * r[0] * r[1]
        I[0, 2] -= m * r[0] * r[2]
        I[1, 2] -= m * r[1] * r[2]

    # Fill symmetric elements
    I[1, 0], I[2, 0], I[2, 1] = I[0, 1], I[0, 2], I[1, 2]

    # Diagonalize the inertia tensor
    eigenvalues, _ = np.linalg.eigh(I)
    Ia, Ib, Ic = np.sort(eigenvalues)

    return Ia, Ib, Ic


def pre_exponential_factor(
  m: float, T_list: list, sigma: int, Ia: float, Ib: float, Ic: float
) -> list:
    """
    Calculate the pre-exponential factor (v) for desorption.
    """
  
    kB = qcel.constants.get("kb")  # Boltzmann constant in J/K
    h = qcel.constants.get("h")  # Planck's constant in J·s
    pi = math.pi

    # Define a helper function to compute v for a single temperature
    def _single_T(
      T: float
      ) -> float:
        """
        Runs the calculation for v of a single temperature value
        """
        # Translational contribution
        translational_part = ((2 * pi * m * kB * T) / h**2)**(3 / 2)

        # Rotational contribution (considering Ia = 0 for linear molecules)
        if Ia == 0:
            rotational_part = (8 * pi**2 * kB * T / h**2) * (Ib / sigma)
        else:
            rotational_part = (pi**0.5 / sigma)*(8 * pi**2 * kB * T / h**2)**(3 / 2) * math.sqrt(Ia * Ib * Ic)

        # Final pre-exponential factor
        return ((kB * T) / h) * translational_part * rotational_part

    return [_single_T(T) for T in T_list]


def main():
    # Call the arguments
    args = parse_arguments()

    # Create a logger
    main_logger = logging.getLogger("preexpt_calc")
    main_logger.setLevel(logging.INFO)

    # File handler for logging to a file
    log_file = (
        "pre_exp_factor_" + args.molecule + "_" + args.level_of_theory + ".log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    main_logger.addHandler(file_handler)

    main_logger.info(welcome_msg)

    #Client from where the xyz will be retrived
    client = ptl.FractalClient(  
        address=args.client_address,
        verify=False,
        username=args.username,
        password=args.password,
    )

    #Name of the molecule and level of theory    
    mol_col = args.molecule_collection
    mol_lot = args.level_of_theory
    temperature_range = args.range_of_temperature.split()
    T_list = list(range(int(temperature_range[0]), int(temperature_range[1]), 1))
    mol = args.molecule

    #Check for collection existence
    check_collection_existence(client, mol_col)

    if mol == None: 
        mol = [mol_col.df.index]
    
    for molecule in mol:
        # Check if all the molecule is optimized at the requested level of theory
        check_optimized_molecule(mol_col, mol_lot, molecule)
        main_logger.info(
            calculation_msg(mol_col, molecule, mol_lot)
        )    
        #Define basic variables of the molecule
        mol_xyz = get_xyz(mol_col,molecule,mol_lot)  
        sym_num = sym_num(mol_xyz)
        symbols, coordinates = parse_coordinates(mol_xyz)
        mol_mass = get_mass(mol_xyz)
    
        #Alings cords with the z axis
        align_coors = align_to_z_axis(symbols, coordinates)
    
        #Calculate moment of inertia and add it to the main log file and an individual one
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coordinates)
        main_logger.info(f"Principal moments of inertia for {molecule} (kg·m²): Ia={Ia:.3e}, Ib={Ib:.3e}, Ic={Ic:.3e}")
        bp_log.setup_logging("I",molecule)
        logger.info(f"{Ia} {Ib} {Ic}")
    
        #Calculate the preexponential factor and add it to the main log file and an individual one
        v = pre_exponential_factor(mol_mass, T_list, sym_num, Ia, Ib, Ic)
        main_logger.info(f"Pre-exponential factor for {molecule} (v): {v:.3e} s⁻¹")
        bp_log.setup_logging("v", molecule)
        logger.info(f"{v}")
        

    if __name__ == "__main__":
        main()
