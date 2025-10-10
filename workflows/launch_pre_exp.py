import sys, time, argparse, logging, os
import math
import numpy as np       # needed by molsym
import pandas as pd	
import qcfractal.interface as ptl
import qcelemental as qcel # needed by molsym
import molsym # for the symmetry point
from typing import Dict, Tuple, List
from beep.utils import logging_utils as bp_log
from qcfractal.interface.collections.optimization_dataset import OptimizationDataset
from qcfractal.interface.client import FractalClient

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
Welcome to the BEEP Range of temperature Pre-Exponential Factor Workflow
---------------------------------------------------------------------------------------


"To deny our impulses is to deny the very thing that makes us human.”

                              – Lana and Lilly Wachowski


---------------------------------------------------------------------------------------

                            By:  Gabriela Silva-Vera  and  Namrata Rani

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
---------------------------------
Starting new calculation for {mol} 
---------------------------------
Collection: {mol_col}
Molecule: {mol}
Level of theory: {level_theory}
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
A command line interface to calculate the pre-exponential factor of a given molecule or a complete collection.
    """
    )
    parser.add_argument(
        "--client-address",
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
        "--molecule-collection",
        default="small_molecules",
        help="The name of the collection containing molecules or radicals (default: small_molecules)",
    )
    parser.add_argument(
        "--level-of-theory",
        default="blyp_def2-svp",
        help="The level of theory in which the molecule is optimized, in the format: method_basis (default: blyp_def2-svp)",
    )          
    parser.add_argument(
        "--range-of-temperature",
        type=int,
        nargs="+",
        default=[10, 273],
        help="Range of temperature in K (default: 10 273)",
    ) 
    parser.add_argument(
        "--temperature-step",
        type=int,
        default=1,
        help="Size of the temperature step on K (default: 1)",
    )
    parser.add_argument(
        "--molecule-surface-area",
        type=float,
        default= 10e-19,
        help="Surface area of each adsorbed molecule, for most molecules is 10e-19 m^(-2) (default: 10e-19)",
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
    client: str, dataset: str, mol_name: str, level_theory: str, collection_type: str = "OptimizationDataset"
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
    Calculates the mass of a molecule in kg using the xyz.
    """
    kg_convert = qcel.constants.get("na")*1000

    mol = qcel.models.Molecule.from_data(xyz)
    m = mol.masses
    mass_sum = m.sum()
    mass = mass_sum/kg_convert
    return(mass)


def get_sym_num(
    xyz: str
) -> int:
    """
    Gives the symmetry number from the xyz using the Molsym package.
    If statements added for linear molecule excemptions
    
    Args:
    - xyz: xyz file (only coordinates, no multiplicity, charge, etc)

    Returns:
    - Symmetry number
    """
    schema = qcel.models.Molecule.from_data(xyz).dict()
    mol = molsym.Molecule.from_schema(schema)
    pg, (paxis, saxis) = molsym.find_point_group(mol)
    if pg == 'D0h':
        return(pg, 2)
    if pg == 'C0v':
        return(pg, 1)
    else:
        s_m = molsym.Symtext.from_molecule(mol).rotational_symmetry_number
        return(pg, s_m)


def parse_coordinates(
  xyz: str
) -> list:
    """
    Parse atomic symbols and their xyz coordinates.
    """
  
    symbols, coordinates = [], []
    for line in xyz.strip().splitlines():
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
    
    kg_convert = qcel.constants.get("na")*1000                                                                                                                                                      
    amu_masses = np.array([qcel.periodictable.to_mass(sym) for sym in symbols])                  
    masses = amu_masses/kg_convert                                                                     
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
  m: float, T_list: list, sigma: int, Ia: float, Ib: float, Ic: float, A: float
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
        translational_part = ((2 * pi * m * kB * T) / h**2)*A

        # Rotational contribution (considering Ia = 0 for linear molecules)
        if Ia == 0:
            rotational_part = (8 * pi**(5/2) * kB * T / h**2) * (Ib / sigma)
        else:
            rotational_part = (pi**0.5 / (sigma * h**3))*(8 * pi**2 * kB * T)**(3 / 2) * math.sqrt(Ia * Ib * Ic)

        # Final pre-exponential factor
        return ((kB * T) / h) * translational_part * rotational_part

    return [_single_T(T) for T in T_list]


def main():
    # Call the arguments
    args = parse_arguments()

    #Client from where the xyz will be retrived
    client = ptl.FractalClient(  
        address=args.client_address,
        verify=False,
        username=args.username,
        password=args.password,
    )

    #Defining different parameters     
    mol_col = args.molecule_collection
    mol_lot = args.level_of_theory
    
    if len(args.range_of_temperature) == 1:
        T_min = T_max = args.range_of_temperature
        T_list = T_min
        T_step = args.temperature_step
    else:
        T_min, T_max = args.range_of_temperature
        T_step = args.temperature_step 
        T_list = list(range(T_min, T_max, T_step))
    mol = args.molecule
    A = args.molecule_surface_area
 
    #Check for collection existence
    ds = client.get_collection("OptimizationDataset", mol_col)
    check_collection_existence(client, mol_col)
    
    # Create a logger
    main_logger = bp_log.setup_logging("log_v", f"{mol_col}")
    main_logger.info(welcome_msg)

    #Create the mol list based on the collection if no argument is given 
    if mol == None:
        mol = ds.df.index
    
    #Create folder for the pre-exponential factor logs
    v_folder = f"./v_{mol_col}"
    os.makedirs(v_folder, exist_ok=True)
  
    for molecule in mol:
        # Check if the molecule is optimized at the requested level of theory
        check_optimized_molecule(ds, mol_lot, molecule)
        main_logger.info(
            calculation_msg(mol_col, molecule, mol_lot)
        )    

        #Define basic variables of the molecule
        mol_xyz = get_xyz(client,mol_col,molecule,mol_lot)  
        point_group, sym_num = get_sym_num(mol_xyz)
        symbols, coordinates = parse_coordinates(mol_xyz)
        mol_mass = get_mass(mol_xyz)
        main_logger.info(f"Point group = {point_group}\nSymmetry number = {sym_num}")

        #Alings cords with the z axis and calculate I
        align_coors = align_to_z_axis(symbols, coordinates)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coordinates)
        main_logger.info(f"Principal moments of inertia for {molecule} (kg·m^(2)): Ia={Ia:.3e}, Ib={Ib:.3e}, Ic={Ic:.3e}")
        
        #Calculate v and create the log file with the information
        v = pre_exponential_factor(mol_mass, T_list, sym_num, Ia, Ib, Ic, A)
        main_logger.info(f"Pre-exponential factor for {molecule} in the range of {T_min}K to {T_max}K with steps of {T_step}K has been calculated")
        
        v_log_file = f"v_{mol_col}/v_{molecule}.dat"
        v_logger = logging.getLogger(f"v_{molecule}")
        v_logger.setLevel(logging.INFO)

        v_handler = logging.FileHandler(v_log_file)
        v_handler.setFormatter(logging.Formatter(" %(message)s"))
        v_logger.addHandler(v_handler)

        table = pd.DataFrame({"T":T_list, "v":v })
        v_logger.info("\n" + table.to_string(index=False))



if __name__ == "__main__":
    main()

