from pathlib import Path
import numpy as np
import random, logging
from typing import List, Tuple
from qcelemental.physical_constants import constants
from qcelemental.models.molecule import Molecule
import qcelemental as qcel


bohr2angst = constants.conversion_factor("bohr", "angstrom")
angst2bohr = constants.conversion_factor("angstrom", "bohr")


def com(geometry: np.ndarray, symbols: list) -> np.ndarray:
    """
    Compute the center of mass for a molecule.

    Parameters:
    - geometry: np.ndarray of shape (N, 3) representing the atomic positions.
    - symbols: list of strings representing the atomic symbols.

    Returns:
    A np.ndarray of shape (3,) representing the center of mass.
    """
    total_mass = 0.0
    com = np.zeros(3)

    for i, symbol in enumerate(symbols):
        atom_mass = qcel.periodictable.to_mass(symbol)
        com += atom_mass * geometry[i]
        total_mass += atom_mass

    com /= total_mass

    return com


def calculate_diameter(cluster_xyz: np.ndarray) -> float:
    """
    Calculate the diameter of a molecule based on its XYZ coordinates.

    Args:
    cluster_xyz (numpy.ndarray): A NumPy array where each row represents an atom and each column represents X, Y, and Z values.

    Returns:
    float: The diameter of the cluster in angstroms.
    """
    # Check if there's only one water molecule in the cluster
    if cluster_xyz.shape == (3, 3):
        return 0.0

    # Calculate pairwise distances between all pairs of atoms in the cluster
    distances = np.linalg.norm(cluster_xyz[:, np.newaxis, :] - cluster_xyz, axis=-1)

    # Set the diagonal elements (self-distances) to a large value to avoid selecting them
    np.fill_diagonal(distances, 0)

    # Find the maximum distance, which represents the diameter of the cluster
    diameter = np.max(distances)

    return diameter


def surface_distance_check(
    cluster: Molecule, mol: Molecule, cut_distance: float
) -> bool:
    """
    Check if any atom in the molecule (mol) is too close to any atom in the cluster based on a specified cutoff distance.

    Parameters:
    - cluster : qcelemental.models.molecule.Molecule
    - mol : qcelemental.models.molecule.Molecule
    - cut_distance : float
        The cutoff distance to determine if atoms are too close. This value is multiplied
        by the global constant 'angst2bohr' to get the actual distance threshold.

    Returns:
    - bool : False if any atom in the molecule is too close to the cluster based on the cutoff distance,
             None otherwise (which evaluates to False in a boolean context).
    """
    for a1 in mol.geometry:
        for a2 in cluster.geometry:
            dis = np.linalg.norm(a1 - a2)
            if dis < cut_distance * angst2bohr:
                return False
    return True


def calculate_displacements(cluster: Molecule, sampling_shell: float) -> (float, float):
    """
    Calculate the minimal and maximal displacements based on the cluster's geometry and a sampling shell.
    The 20%  most distance atoms from the origin are considered for calculating the minimum distance.

    Parameters:
    - cluster : qcelemental.models.molecule.Molecule
        Molecular cluster for which displacements are calculated.
    - sampling_shell : float
        Size of the shell to use for sampling.

    Returns:
    - tuple(float, float) : Minimum and maximum displacements.
    """
    atms_prctg = 20
    norms = [np.linalg.norm(i) for i in cluster.geometry]
    # Get the top 20% distances and use the average as minimal sampling distance
    num_top = max(1, int(len(norms) * atms_prctg / 100))
    top_norms = np.partition(norms, -num_top)[-num_top:]
    dis_min = np.mean(top_norms)
    dis_max = dis_min + sampling_shell * angst2bohr

    return dis_min, dis_max


def generate_shift_vector(dis_min: float, dis_max: float) -> np.ndarray:
    """
    Generate a random shift vector with a magnitude between the given minimum and maximum distances.

    Parameters:
    - dis_min : float
        Minimum distance for the shift vector.
    - dis_max : float
        Maximum distance for the shift vector.

    Returns:
    - np.ndarray : Randomly generated shift vector.
    """
    vector = (
        np.random.random_sample((3,)) - 0.5
    )  # shift to [-0.5, 0.5) for better isotropy
    unit_vector = vector / np.linalg.norm(vector)
    random_radius = np.random.uniform(dis_min, dis_max)
    shift_vector = unit_vector * random_radius
    # logger.debug(f"Generated the follwing vector: {scaled_vector} within the radius {random_radius}")
    return shift_vector


def create_molecule(cluster: Molecule, mol_shift: Molecule) -> Molecule:
    """
    Create a new molecule by combining a cluster and a shifted molecule.

    Parameters:
    - cluster : qcelemental.models.molecule.Molecule
        Molecular cluster to be combined.
    - mol_shift : qcelemental.models.molecule.Molecule
        Shifted molecule to be combined with the cluster.

    Returns:
    - qcelemental.models.molecule.Molecule : Combined molecule.
    """
    atms = []
    atms.extend(list(cluster.symbols))
    atms.extend(list(mol_shift.symbols))

    geom = []
    geom.extend(list(cluster.geometry.flatten()))
    geom.extend(list(mol_shift.geometry.flatten()))

    return Molecule(symbols=atms, geometry=geom, fix_com=False, fix_orientation=False)


attempts = 0


def random_molecule_sampler(
    cluster: Molecule,
    target_molecule: Molecule,
    sampling_shell: float,
    debug: bool = False,
) -> Tuple[List[Molecule], Molecule]:
    """
    Sample random molecule placements around a given molecular cluster.

    Parameters:
    - cluster : qcelemental.models.molecule.Molecule
        Molecular cluster around which the target molecule will be sampled.
    - target_molecule : qcelemental.models.molecule.Molecule
        Molecule to be sampled around the cluster.
    - number_of_structures : int (default=10)
        Number of sampled structures to generate.
    - sampling_shell : float (default=2.0)
        Size of the shell to use for sampling.
    - debug : bool (default=False)
        If True, additional debugging information is provided.

    Returns:
    - list of qcelemental.models.molecule.Molecule : List of sampled molecular structures.
    - qcelemental.models.molecule.Molecule : Debug molecule (if debug=True).
    """
    log_level = logging.INFO
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    logger.info("Welcome to the molecule sampler!")
    logger.info(f"Author: svogt")
    logger.info(f"Date: 10/24/2020")
    logger.info(f"Version: 0.2.1")
    logger.info(f"Cluster to be sampled: {cluster}")
    logger.info(f"Sampled molecule: {target_molecule}")
    logger.info(f"Size of the sampling shell: {sampling_shell}")

    dis_min, dis_max = calculate_displacements(cluster, sampling_shell)
    target_mol_diam = calculate_diameter(target_molecule.geometry)
    cluster_diam = calculate_diameter(cluster.geometry)

    # initialize variables
    cluster_with_sampled_mol = []
    sampled_mol = []
    debug_molecule = None
    c = 0
    attempts = 0
    binding_site_size = 3
    atoms_per_cluster_mol = 3
    total_attempts = 500
    surface_closness_cutoff = 1.52  # Angstrom vdW radius of Oxygen

    max_structures = max(3, (len(cluster.symbols) / atoms_per_cluster_mol) // 3)
    logger.info(f"Maximum number of structures to be sampled: {max_structures }")
    fill_num = len(str(max_structures))

    while c < max_structures:
        attempts += 1
        if attempts == total_attempts:
            break

        new_s_num = str(c).zfill(fill_num)
        shift_vect = generate_shift_vector(dis_min, dis_max)
        norm = np.linalg.norm(np.array(shift_vect))

        mol_shift = target_molecule.scramble(
            do_shift=shift_vect, do_rotate=True, do_resort=False, deflection=1.0
        )[0]

        skip_remaining = False

        # Making sure initial structures are not too close to each other
        if sampled_mol:
            close_condition = target_mol_diam
            logger.debug(f"The closesness cutoff is: {close_condition * bohr2angst}")
            for m in sampled_mol:
                dis_vec = com(mol_shift.geometry, mol_shift.symbols) - com(
                    m.geometry, m.symbols
                )
                if np.linalg.norm(dis_vec) < close_condition:  # * angst2bohr:
                    skip_remaining = True
                    break
            if skip_remaining == True:
                continue

        # Check if any two atoms of the sampled molecule and cluster are not closer than a given distance
        logger.debug(
            f"The closesness to the surface cutoff is: {surface_closness_cutoff * bohr2angst}"
        )
        if not surface_distance_check(cluster, mol_shift, surface_closness_cutoff):
            continue

        logger.debug(f"Generated the molecule {new_s_num}:")
        logger.debug(f"Displacement vector: {shift_vect * bohr2angst}")
        logger.debug(f"Norm of the displacement vector: {norm * bohr2angst}")

        # create new sampled molecule + cluster
        sampled_mol.append(mol_shift)
        molecule = create_molecule(cluster, mol_shift)
        cluster_with_sampled_mol.append(molecule)

        if debug:
            if not debug_molecule:
                debug_molecule = molecule
            else:
                debug_molecule = create_molecule(debug_molecule, mol_shift)
        c += 1

    final_num_struc = len(cluster_with_sampled_mol)
    logger.info(f"Number of generated initial structures: {final_num_struc} ")

    return cluster_with_sampled_mol, debug_molecule


def single_site_spherical_sampling(
    cluster,
    sampling_mol,
    sampled_mol_size,
    sampling_shell,
    grid_size,
    purge,
    noise,
    zenith_angle,
    print_out=True,
):
    out_string = """
                  Welcome to the single site molecule sampler!

   Author: svera, svogt
   Date:   12/12/2022
   Version: 0.1.0

   Cluster to sampled: {}
   Sampling molecule : {}
   Grid to be used: {}
   Size of the sampling shell: {}

   """.format(
        cluster, sampling_mol.name, grid_size, sampling_shell
    )

    bohr2angst = constants.conversion_factor("bohr", "angstrom")
    angst2bohr = constants.conversion_factor("angstrom", "bohr")

    def _Ry(rad):
        """Rotation matrix around y axis"""
        return np.array(
            [
                [np.cos(rad), 0.0, np.sin(rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(rad), 0.0, np.cos(rad)],
            ]
        )

    def _Rx(rad):
        """Rotation matrix around x axis"""
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rad), -np.sin(rad)],
                [0.0, np.sin(rad), np.cos(rad)],
            ]
        )

    def _Rz(rad):
        """Rotation matrix around z axis"""
        return np.array(
            [
                [np.cos(rad), -np.sin(rad), 0.0],
                [np.sin(rad), np.cos(rad), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def _com(x, y):
        """Center of mass. x is geom and y the symbols"""
        num = 0
        den = 0

        cont = 0
        for i in x:
            num = num + (qcel.periodictable.to_mass(y[cont]) * i)
            den = den + qcel.periodictable.to_mass(y[cont])
            cont += 1

        com = num / den
        return com

    if grid_size == "normal":
        grid = (4, 12)
    if grid_size == "sparse":
        grid = (3, 6)
    if grid_size == "dense":
        grid = (5, 16)

    # Generate the spherical grid
    radio = sampling_shell * angst2bohr

    phi_end = zenith_angle
    phi_interval = phi_end / (2 * grid[0])

    theta_end = 2 * np.pi
    theta_interval = theta_end / (1.5 * grid[1])

    theta = np.linspace(0, theta_end, grid[0])
    phi = np.linspace(0, phi_end, grid[1])

    phi = np.linspace(0, phi_end, grid[0], endpoint=False)
    theta = np.linspace(0, theta_end, grid[1], endpoint=False)

    # Grid
    grid_xyz_i = [[0, 0, radio]]

    for i in theta:
        for n in phi[1:]:
            grid_p = []

            r = radio

            if noise:
                r += r * random.random() / 2.0
                n += random.uniform(-phi_interval, phi_interval)
                i += random.uniform(-theta_interval, theta_interval)

            x = r * np.sin(n) * np.cos(i)
            y = r * np.sin(n) * np.sin(i)
            z = r * np.cos(n)

            grid_p.append(x)  # * bohr2angst)
            grid_p.append(y)  # * bohr2angst)
            grid_p.append(z)  # * bohr2angst)

            grid_xyz_i.append(grid_p)

    # purge

    grid_xyz = grid_xyz_i.copy()
    if purge:
        purge = purge * angst2bohr
        remove_list = []
        for i in range(0, len(grid_xyz)):
            for j in range(i + 1, len(grid_xyz)):
                gridp_dis = np.linalg.norm(
                    np.array(grid_xyz_i[i]) - np.array(grid_xyz_i[j])
                )
                if gridp_dis < purge:
                    remove_list.append(grid_xyz_i[j])
                    print("removing point")
        for n in remove_list:
            try:
                grid_xyz.remove(n)
            except ValueError:
                continue

    out_string += """Total grid points: {}
""".format(
        str(len(grid_xyz))
    )

    # target

    # Define unit vectors
    vz = np.array([0.0, 0.0, 1.0])
    vy = np.array([0.0, 1.0, 0.0])
    vx = np.array([1.0, 0.0, 0.0])

    # Get center of mass of target molecule
    len_target = int(sampled_mol_size)
    target_mol_geom = cluster.geometry[-len_target:, :]
    target_mol_symb = cluster.symbols[-len_target:]
    com_target = _com(target_mol_geom, target_mol_symb)

    # Shift to set the origin at the center of mass of target molecule
    cluster_tras = cluster.scramble(
        do_shift=-com_target, do_rotate=False, do_resort=False
    )[0]

    # Get geometries of shifted cluster
    cluster_geom = cluster_tras.geometry
    cluster_symb = cluster_tras.symbols

    # Ensure that the COM is in the first quadrant
    cluster_geom_s = []
    for i in cluster_geom:
        new_coords = np.multiply(i, np.sign(-com_target))
        cluster_geom_s.append(new_coords)

    com_refl = _com(cluster_geom_s, cluster_symb)

    # Define rotation angles
    theta_rot_rz = np.arccos(
        np.dot(com_refl[:-1], vy[:-1]) / np.linalg.norm(com_refl[:-1])
    )
    out_string += """Rotation angle around the z axis (xy plane): {}
""".format(
        str(theta_rot_rz * 180.0 / np.pi)
    )

    cluster_rot_geom_rz = []
    for i in cluster_geom_s:
        cluster_rot_geom_rz.append(np.dot(_Rz(theta_rot_rz), i))

    v_check_1 = np.dot(_Rz(theta_rot_rz), -com_target)

    out_string += """Rotation check around the z axis: {}
""".format(
        v_check_1
    )

    com_int = _com(cluster_rot_geom_rz, cluster_symb)

    # Define rotation angle around x axis
    theta_rot_rx = np.arccos(np.dot(com_int[1:], vz[1:]) / np.linalg.norm(com_int[1:]))

    cluster_rot_geom = []
    for i in cluster_rot_geom_rz:
        new_coords = np.dot(_Rx(theta_rot_rx), i)
        cluster_rot_geom.append(np.multiply(new_coords, np.array([1.0, 1.0, -1.0])))

    v_check_2 = np.dot(_Rx(theta_rot_rx), v_check_1)

    out_string += """Rotation check for rotation around the x axis: {}
""".format(
        v_check_2
    )

    com_final = _com(cluster_rot_geom, cluster_symb)

    theta_check = np.arccos(np.dot(com_final, vz) / np.linalg.norm(com_final))
    out_string += """Final angle check (should be 180.0): {}
""".format(
        str(theta_check * 180.0 / np.pi)
    )

    # prepare sampling molecule (molecule that will sample the target binding site)
    sampling_mol_com = _com(sampling_mol.geometry, sampling_mol.symbols)
    sampling_mol = sampling_mol.scramble(
        do_shift=-1 * sampling_mol_com, do_rotate=False, do_resort=False
    )[0]

    # generated estructures
    molecules = []

    vis_mol_geom = list(np.array(cluster_rot_geom).flatten())
    vis_mol_atms = list(cluster_symb)

    # generate the new structures
    for i in grid_xyz:
        # move the center of mass of sampled molecule to the point i in grid
        shift_vector = np.array(i) * angst2bohr
        sampling_final_mol = sampling_mol.scramble(
            do_shift=shift_vector, do_rotate=True, do_resort=False
        )[0]

        sampling_geom = sampling_final_mol.geometry
        sampling_symb = sampling_final_mol.symbols

        # create the new structure
        atms = list(cluster_symb)
        sampling_atms = list(sampling_symb)
        atms.extend(sampling_atms)
        vis_mol_atms.extend("H")

        geom = list(np.array(cluster_rot_geom).flatten())
        sampling_geom = list(np.array(sampling_geom).flatten())
        geom.extend(sampling_geom)
        vis_mol_geom.extend(list(i))

        molecule = qcel.models.Molecule(
            symbols=atms, geometry=geom, fix_com=True, fix_orientation=True
        )
        molecules.append(molecule)

    # vis_mol_atms.extend("B")
    # vis_mol_geom.extend([0.0, 0.0, -30.0])
    # vis_mol_atms.extend("B")
    # vis_mol_geom.extend([0.0, 0.0, 22.0])

    all_mols = qcel.models.Molecule(
        symbols=vis_mol_atms, geometry=vis_mol_geom, fix_com=True, fix_orientation=True
    )

    if print_out:
        print(out_string)

    return molecules  # , all_mols


if __name__ == "__main__":
    main()


# def main():
#    from optparse import OptionParser
#
#    parser = OptionParser()
#    parser.add_option(
#        "-w", "--water_cluster", dest="c_mol", help="The name of the water cluster"
#    )
#    parser.add_option(
#        "-m",
#        "--molecule",
#        dest="s_mol",
#        help="The name of the molecule to be sampled (from the small_mol collection",
#    )
#    parser.add_option(
#        "-n",
#        "--number_of_structures",
#        dest="s_num",
#        type="int",
#        help="The number of initial structures to be created (Default = 10)",
#        default=10,
#    )
#    parser.add_option(
#        "-s",
#        "--sampling_shell",
#        dest="sampling_shell",
#        type="float",
#        default=1.5,
#        help="The shell size of sampling space (Default = 1.5 Angstrom)",
#    )
#    parser.add_option(
#        "--xyz-path",
#        dest="xyz_path",
#        default=None,
#        help="The path to save the xyz files, if non is provided this will be omitted",
#    )
#    parser.add_option(
#        "--print_out", action="store_true", dest="print_out", help="Print an output"
#    )
#
#    options = parser.parse_args()[0]
#
#    molecule_sampler(
#        options.c_mol,
#        options.s_mol,
#        number_of_structures=options.s_num,
#        sampling_shell=options.sampling_shell,
#        # save_xyz=options.xyz_path,
#        print_out=options.print_out,
#    )
#
#    return
