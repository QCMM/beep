import numpy as np
from qcelemental.physical_constants import constants
import qcelemental as qcel
from pathlib import Path
import random


def random_molecule_sampler(
    cluster,
    target_molecule,
    number_of_structures=10,
    sampling_shell=2.5,
    # save_xyz=[],
    print_out=False,
):
    out_string = """
                   Welcome to the molecule sampler!

    Author: svogt
    Date:   10/24/2020
    Version: 0.1.2

    Cluster to sampled: {}
    Sampled molecule : {}
    Number of structures to be generated: {}
    Size of the sampling shell: {}

    """.format(
        cluster, target_molecule, number_of_structures, sampling_shell
    )

    bohr2angst = constants.conversion_factor("bohr", "angstrom")
    angst2bohr = constants.conversion_factor("angstrom", "bohr")

    fill_num = len(str(number_of_structures))

    # Define the maximum  and minimum displacements
    dis_min = max([np.linalg.norm(i) for i in cluster.geometry])  # remove 0.75
    dis_max = dis_min + sampling_shell * angst2bohr

    out_string += """
    Fixing the maximum and minimums distances for the sampling space:

    Minimum distance = {} Angstrom
    Maximum distance = {} Angstrom

    """.format(
        dis_min * bohr2angst, dis_max * bohr2angst
    )

    # Creating the total geomtry sampling list
    molecules = []
    atms_sm = []
    atms_sm.extend(list(cluster.symbols))
    geom_sm = []
    geom_sm.extend(list(cluster.geometry.flatten()))

    c = 0

    out_string += """Commencing the creation of the new structures... 
    """
    while c < number_of_structures:
        # Sample between:  a<dis<b (b - a) * random_sample() + a
        # Generate random shift vector
        shift_vect = (dis_max + dis_max) * np.random.random_sample((3,)) - dis_max
        norm = np.linalg.norm(np.array(shift_vect))

        # Check if shift vector is within the defined range
        if not ((norm < dis_max) and (norm > dis_min)):
            continue

        # Shift, rotate and
        mol_shift = target_molecule.scramble(
            do_shift=shift_vect, do_rotate=True, do_resort=False, deflection=1.0
        )[0]

        new_s_num = str(c).zfill(fill_num)

        # Creating list with atoms of the joined molecule
        atms = []
        atms.extend(list(cluster.symbols))
        atms.extend(list(mol_shift.symbols))
        # Creating list with geometry of the joined molecule
        geom = []
        geom.extend(list(cluster.geometry.flatten()))
        geom.extend(list(mol_shift.geometry.flatten()))
        # Creating the new molecule
        molecule = qcel.models.Molecule(
            symbols=atms,
            geometry=geom,
            fix_com=False,
            fix_orientation=False,
        )
        if save_xyz:
            molecule.to_file(save_xyz + "/st_" + str(new_s_num) + ".xyz")
        molecules.append(molecule)
        # Creating molecule with all the displaced molecules
        atms_sm.extend(list(mol_shift.symbols))
        geom_sm.extend(list(mol_shift.geometry.flatten()))
        c += 1
        out_string += """
        ---------------------------------------------------------------------- 
        Generated the molecule {}: 

        Displacement vector: {}
        Norm of the displacement vector: {}
        ----------------------------------------------------------------------        

        """.format(
            new_s_num, shift_vect * bohr2angst, norm * bohr2angst
        )

    molecules_shifted = qcel.models.Molecule(
        symbols=atms_sm,
        geometry=geom_sm,
        fix_com=False,
        fix_orientation=False,
        validated=False,
    )
    # if save_xyz:
    #    molecules_shifted.to_file(save_xyz + "/all.xyz")
    # out_string += """ Thank you for using Molecule sampling! """
    # if print_out:
    #    print(out_string)
    return molecules


def single_site_spherical_sampling(
    cluster,  # cluster + target_molecule
    sampling_mol,  # sampled_molecule
    sampled_mol_size,  # number of atoms of target molecule
    sampling_shell=2.5,
    grid_size="normal",
    save_xyz=[],
    purge=0.5,  
    noise=False,
    zenith_angle=np.pi / 2,
    print_out=False,
):

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
    if grid_size == "tight":
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
    theta = np.linspace(
        0, theta_end, grid[1], endpoint=False
    )

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
        remove_list = []
        for i in range(0, len(grid_xyz)):
            for j in range(i+1, len(grid_xyz)):
                gridp_dis = np.linalg.norm(np.array(grid_xyz_i[i]) - np.array(grid_xyz_i[j]))
                if gridp_dis < purge:
                    remove_list.append(grid_xyz_i[j])
                    print("removing point")
        for n in remove_list:
            grid_xyz.remove(n)

    print("Total grid points: ", len(grid_xyz))

    #cluster = client.query_molecules(cluster_id)[0]
    #sampled = client.query_molecules(sampled_id)[0]

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

    print("Original center of mass of  cluster:  ", -com_target)
    # Shift to set the origin at the center of mass of target molecule
    cluster_tras = cluster.scramble(
        do_shift=-com_target, do_rotate=False, do_resort=False
    )[0]

    # Get geometries of shifted cluster
    cluster_geom = cluster_tras.geometry
    cluster_symb = cluster_tras.symbols

    cluster_geom_s = []
    for i in cluster_geom:
        new_coords = np.multiply(i, np.sign(-com_target))
        cluster_geom_s.append(new_coords)

    com_refl = _com(cluster_geom_s, cluster_symb)

    print("center of mass of adjusted cluster: ", com_refl)

    # Define rotation angles
    theta_rot_rz = np.arccos(
        np.dot(com_refl[:-1], vy[:-1]) / np.linalg.norm(com_refl[:-1])
    )
    print("Rotation angle around the z axis (xy plane)", theta_rot_rz * 180.0 / np.pi)

    cluster_rot_geom_rz = []
    for i in cluster_geom_s:
        cluster_rot_geom_rz.append(np.dot(_Rz(theta_rot_rz), i))

    v_check_1 = np.dot(_Rz(theta_rot_rz), -com_target)

    print("Rotation check: ", v_check_1)

    com_int = _com(cluster_rot_geom_rz, cluster_symb)
    print("Intermediate center of mass:", com_int)

    # Define rotation angle around x axis
    theta_rot_rx = np.arccos(np.dot(com_int[1:], vz[1:]) / np.linalg.norm(com_int[1:]))
    print("Rotation angle around the x axis (yz plane)", theta_rot_rx * 180.0 / np.pi)

    cluster_rot_geom = []
    for i in cluster_rot_geom_rz:
        new_coords = np.dot(_Rx(theta_rot_rx), i)
        cluster_rot_geom.append(np.multiply(new_coords, np.array([1.0, 1.0, -1.0])))

    v_check_2 = np.dot(_Rx(theta_rot_rx), v_check_1)

    print("Rotation check: ", v_check_2)

    com_final = _com(cluster_rot_geom, cluster_symb)
    print("Final center of mass:", com_final)

    theta_check = np.arccos(np.dot(com_final, vz) / np.linalg.norm(com_final))
    print("Final angle: ", theta_check * 180.0 / np.pi)

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

    return molecules#, all_mols


if __name__ == "__main__":
    main()


def main():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option(
        "-w", "--water_cluster", dest="c_mol", help="The name of the water cluster"
    )
    parser.add_option(
        "-m",
        "--molecule",
        dest="s_mol",
        help="The name of the molecule to be sampled (from the small_mol collection",
    )
    parser.add_option(
        "-n",
        "--number_of_structures",
        dest="s_num",
        type="int",
        help="The number of initial structures to be created (Default = 10)",
        default=10,
    )
    parser.add_option(
        "-s",
        "--sampling_shell",
        dest="sampling_shell",
        type="float",
        default=1.5,
        help="The shell size of sampling space (Default = 1.5 Angstrom)",
    )
    parser.add_option(
        "--xyz-path",
        dest="xyz_path",
        default=None,
        help="The path to save the xyz files, if non is provided this will be omitted",
    )
    parser.add_option(
        "--print_out", action="store_true", dest="print_out", help="Print an output"
    )

    options = parser.parse_args()[0]

    molecule_sampler(
        options.c_mol,
        options.s_mol,
        number_of_structures=options.s_num,
        sampling_shell=options.sampling_shell,
        # save_xyz=options.xyz_path,
        print_out=options.print_out,
    )

    return
