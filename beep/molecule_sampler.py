import numpy as np
from qcelemental.physical_constants import constants
import qcelemental as qcel
from pathlib import Path


def molecule_sampler(
    w_molecule,
    s_molecule,
    number_of_structures=10,
    sampling_shell=1.5,
    save_xyz=[],
    print_out=False,
):
    out_string = """
                   Welcome to the molecule sampler!

    Author: svogt
    Date:   10/24/2020
    Version: 0.1

    Number of structures to be generated: {}
    Size of the sampling shell: {}
    Folder to save the xyz files of the generated molecule: {}

    """.format(
        w_molecule, s_molecule, number_of_structures, sampling_shell, save_xyz
    )

    bohr2angst = constants.conversion_factor("bohr", "angstrom")
    angst2bohr = constants.conversion_factor("angstrom", "bohr")

    fill_num = len(str(number_of_structures))


    # Define the maximum  and minimum displacements
    dis_min = max([np.linalg.norm(i) for i in w_molecule.geometry])  # remove 0.75
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
    atms_sm.extend(list(w_molecule.symbols))
    geom_sm = []
    geom_sm.extend(list(w_molecule.geometry.flatten()))

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
        mol_shift = s_molecule.scramble(
            do_shift=shift_vect, do_rotate=True, do_resort=False, deflection=1.0
        )[0]

        new_s_num = str(c).zfill(fill_num)

        # Creating list with atoms of the joined molecule
        atms = []
        atms.extend(list(w_molecule.symbols))
        atms.extend(list(mol_shift.symbols))
        # Creating list with geometry of the joined molecule
        geom = []
        geom.extend(list(w_molecule.geometry.flatten()))
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
    if save_xyz:
        molecules_shifted.to_file(save_xyz + "/all.xyz")
    out_string += """ Thank you for using Molecule sampling! """
    if print_out:
        print(out_string)
    return molecules


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
        save_xyz=options.xyz_path,
        print_out=options.print_out,
    )

    return


if __name__ == "__main__":
    main()
