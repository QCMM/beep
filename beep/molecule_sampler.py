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
    #save_xyz=[],
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
    #if save_xyz:
    #    molecules_shifted.to_file(save_xyz + "/all.xyz")
    #out_string += """ Thank you for using Molecule sampling! """
    #if print_out:
    #    print(out_string)
    return molecules

def single_mol_spherical_sampling(
    cluster
    target_molecule,
    water_cluster_size=22,
    sampling_radius=2.5,
    grid_size = "normal"
    purge = 0.01 # None if no purging is desired
    noise = True
    print_out=False,
):

    if gird_size = "normal":
        gird = (10,10)
    if gird_size = "sparse":
        gird = (5,5)
    if gird_size = "tight":
        gird = (20,20)

    phi_end = np.pi/2
    phi_interval = phi_end/gird[0]
    phi_noise = []

    theta_end = 2 * np.pi
    theta_interval = theta_end/grid[1]
    theta_noise = []

    for i in range(grid[0])
        phi_noise.append(random.uniform(-phi_interval, phi_interval))
    for i in range(grid[1])
        theta_noise.append(random.uniform(-theta_interval, theta_interval))

    phi = np.linspace(0, phi_end, grid[0], endpoint=False) + np.array(phi_noise)
    theta = np.linspace(0, theta_end, grid[1], endpoint=False) + np.array(theta_noise)

    theta, phi = np.meshgrid(theta, phi)
    grid_xyz = qcel.bohr2angstrom * sampling_shell * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]

    if purge:
        for i in range(len(grid_xyz)):
            for j in (i, len(grid_xyz)):
                if np.linalg.norm(grid_xyz[i] -gird_xyz[j]) <= purge: 
                    grid_xyz.remove(grid_xyz[j])

     
    len_target = len(target_molecule.symbols)

    target_mol_geom = cluster.geometry()[-len_target:,:]
    target_mol_sym = cluster.symbols()[-len_target:]

    com_target = com(target_mol_geom, target_mol_symb)

    cluster_tras = cluster.scramble(do_shift=-1*com_target, do_rotate=False, do_resort=False)[0]

    x,y,z = com_target
    theta_com = np.arctan((np.sqrt(x**2 + y**2))/z)

    def _Ry(rad):
        return np.array([[np.cos(rad), 0.0, np.sin(rad)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(rad), 0.0, np.cos(rad)]])


    cluster_geom = cluster_tras_def.geometry
    cluster_sym = cluster_tras_def.symbols

    cluster_rot_geom = []

    for i in cluster_tras.geometry:
        cluster_rot_geom.append(np.dot(Ry(theta_centro), i))
   

    #analogo para el formil pero sin la rotación
    sampled_molecule_len = len(sampled.geometry) #numero de atomos

    sampled_com = centro_masas(sampled.geometry, sampled.symbols)

    sampled_mol = sampled.scramble(do_shift=-1* sampled_com, do_rotate=False, do_resort=False)[0]

    #numero de estructuras creadas
    num = 0

    #guardar todas las estructuras generadas
    molecules = []

    #loop generador de estructuras
    for i in grid_xyz:
        #posicionar el centro de masas en la posición que necesito
        shift_vector = np.array(i)*angst2bohr
        sampled_final_mol = sampled_mol.scramble(do_shift=shift_vector, do_rotate=True, do_resort=False)[0]

        #geometria y símbolos radical
        sampled_geom = sampled_final_mol.geometry
        sampled_symb = sampled_final_mol.symbols

        #uniendo toda la información de las moléculas hasta ahora
        atms = list(cluster_symb)
        sampled_atms = list(sampled_symb)

        atms.extend(sampled_atms)

        geom = list(np.array(cluster_rot_geom).flatten())
        sampled_geom = list(np.array(sampled_geom).flatten())

        geom.extend(radical_geom)

        molecule = qcel.models.Molecule(symbols=atms,
                                        geometry=geom,
                                        fix_com=True,
                                        fix_orientation=True)

        molecules.append(molecule)


    return molecules

















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
        #save_xyz=options.xyz_path,
        print_out=options.print_out,
    )

    return

