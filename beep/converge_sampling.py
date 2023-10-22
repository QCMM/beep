import sys, time
from .molecule_sampler import random_molecule_sampler as mol_sample
from .molecule_sampler import single_site_spherical_sampling as single_site_mol_sample
import qcfractal.interface as ptl
import numpy as np
from pathlib import Path
from optparse import OptionParser

def generate_shell_list(sampling_shell, condition):
    # For sparse
    if condition == 'sparse':
        return [sampling_shell]

    # For normal
    elif condition == 'normal':
        return [sampling_shell, sampling_shell * 0.8, sampling_shell * 1.2]

    # For fine
    elif condition == 'fine':
        return [sampling_shell, sampling_shell * 0.75, sampling_shell * 0.9, sampling_shell * 1.1, sampling_shell * 1.5]

    else:
        raise ValueError("Condition should be one of ['sparse', 'normal', 'fine']")


def sampling(
    method,
    basis,
    program,
    tag,
    kw_id,
    opt_dset_name,
    opt_lot,
    rmsd_symm,
    rmsd_val,
    target_mol,
    cluster,
    o_file,
    client,
    sampling_shell,
    sampling_condition,
    sampled_mol_size=None,
    water_cluster_size=22,
    grid_size="sparse",
    purge=None,
    noise=False,
    zenith_angle=np.pi / 2,
    single_site=False,
):
    def print_out(string):
        with open(o_file, "a") as f:
            f.write(string)

    smpl_opt_dset_name = "pre_" + str(opt_dset_name)

    print_out(
        """
    Water cluster: {}
    Small molecule: {}
    method: {}
    basis:  {}
    """.format(
            cluster, target_mol, method, basis
        )
    )

    frequency = 200
    out_string = ""

    try:
        smpl_ds_opt = client.get_collection("OptimizationDataset", smpl_opt_dset_name)
        out_string += """OptimizationDateset {} already exists, new sampled structures will be saved here.
        """.format(
            smpl_opt_dset_name
        )
    except KeyError:
        smpl_ds_opt = ptl.collections.OptimizationDataset(
            smpl_opt_dset_name, client=client
        )
        smpl_ds_opt.save()
        out_string += """Creating new OptimizationDataset {} for optimizations from the sampling.
        """.format(
            smpl_opt_dset_name
        )

    try:
        ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
        out_string += """OptimizationDateset {} already exists, new unique structures will be saved here.
        """.format(
            opt_dset_name
        )
    except KeyError:
        ds_opt = ptl.collections.OptimizationDataset(opt_dset_name, client=client)
        ds_opt.save()
        out_string += """Creating new OptimizationDataset {}. New unique structures will be stored here.
        """.format(
            opt_dset_name
        )

    if program == "terachem":
        if len(method.split("-")) == 2:
            method = method.split("-")[0]

    # DFT dftd3=true terachem kw_id = ?
    spec = {
        "name": opt_lot,
        "description": "Geometric Optimziation ",
        "optimization_spec": {"program": "geometric", "keywords": {"maxiter": 150}},
        "qc_spec": {
            "driver": "gradient",
            "method": method,
            "basis": basis,
            "keywords": kw_id,
            "program": program,
        },
    }

    try:
        smpl_ds_opt.add_specification(**spec)
        out_string += """Adding the specification {} to the {} OptimizationData set.

        """.format(
            spec["name"], smpl_opt_dset_name
        )
    except KeyError:
        out_string += """The specification {} is already present in the {} OptimizationData set! Nothing to do here.

        """.format(
            spec["name"], smpl_opt_dset_name
        )
    out_string += """

    QC data:
    Program: {}
    Method: {}
    Basis:  {}

    Starting Convergence procedure....

    """.format(
        program, method, basis
    )

    print_out(out_string)
    print(out_string)

    c = 1
    converged = False
    shell_list = generate_shell_list(sampling_shell, sampling_condition)
    nmol = 0
    # while not converged:
    for shell in shell_list:
        out_string = ""
        mol_l = " "
        entry_list = []
        complete_opt_name = []

        if not single_site:
            molecules, _ = mol_sample(
                cluster,
                target_mol,
                sampling_shell=shell,
                debug=True,
            )
        else:
            molecules = single_site_mol_sample(
                cluster=cluster,
                sampling_mol=target_mol,
                sampled_mol_size=sampled_mol_size,
                sampling_shell=shell,
                grid_size=grid_size,
                purge=purge,
                noise=noise,
                zenith_angle=zenith_angle,
                print_out=True,
            )

        for m in molecules:
            nmol += 1
            entry_name = opt_dset_name + "_" + str(nmol).zfill(4)
            entry_list.append(entry_name)
            try:
                smpl_ds_opt.add_entry(entry_name, m, save=True)
            except KeyError as e:
                print_out(str(e) + "\n")

            mol_l += str(m.get_hash()) + "\n"

        print_out(
            """Initiating round {}: Generated {} structures to be optimized:
    Molecule Hash:
    {}
    
        """.format(
                c, len(molecules), mol_l
            )
        )

        smpl_ds_opt.compute(opt_lot, tag=tag)
        print_out("The optimization procedures were submitted!")
        pid = ""
        for n in entry_list:
            opt_rec = smpl_ds_opt.get_record(n, opt_lot)
            pid += str(opt_rec.id) + " "

        print_out(
            """ The procedure IDs of round {} molecules are:
    {}
    
        """.format(
                c, pid
            )
        )
        jobs_complete = False
        print_out("Waiting for optimizations to complete\n\n")

        # Checks if no more jobs are running

        while not jobs_complete:
            status = []
            for i in pid.split():
                rr = client.query_procedures(int(i))[0]
                status.append(rr.status)

            # Initaial spec Query to avoid status bug
            smpl_ds_opt = client.get_collection(
                "OptimizationDataset", smpl_opt_dset_name
            )
            smpl_ds_opt.query(opt_lot)

            if "INCOMPLETE" in status:
                print_out("Some jobs are still running, will sleep now\n")
                time.sleep(frequency)

            if not "INCOMPLETE" in status:
                print_out("All jobs finished!\n")
                jobs_complete = True

        # Gets the records for the completed jobs
        for n in entry_list:
            opt_rec = smpl_ds_opt.get_record(n, opt_lot)
            if opt_rec.status == "COMPLETE":
                complete_opt_name.append(n)

        new = 0
        # Adding unique structures to opt_dset_name
        print_out(
            """
    Starting to populate the OptimizationDataset: {}
    
    In the first run the dataset will be seeded with the first structures and from there one every additional
    structure will be compared with all of the existing structures""".format(
                opt_dset_name
            )
        )

        print_out(
            """
    
    Names of newly optimized molecules: {}
    ************************** Starting  RMSD Filtering procedure ************************** 
    
    """.format(
                complete_opt_name
            )
        )

        for n in complete_opt_name:
            out_string = ""
            i = smpl_ds_opt.get_record(n, opt_lot).id
            record = client.query_procedures(id=i)[0]
            final_mol = record.get_final_molecule()

            # Adding first optimized molecule to the opt_dset_name
            if not ds_opt.data.records:
                print_out(
                    "Seeding the OptimizationDataset with the first structure {} \n".format(
                        n
                    )
                )
                ds_opt.add_entry(n, final_mol, save=True)
                continue

            # print_out(out_string)
            # Comparing molecules to already existing molecules in the Optopt_dset_name, only adding them if RMSD > X
            id_list = ds_opt.data.records.items()
            out_string += """
    
    --------------------------------------------------------------------------------------------------
    Starting the comparison of molecule {} to the existing molecules in the opt_dset_name (unique molecules)
    The RMSD criteria is: {}
    
    """.format(
                n, rmsd_val
            )
            unique = []
            for j in id_list:
                # The initial molecule in the OptDataset is the optimized molecule
                rj = client.query_results(
                    molecule=j[1].initial_molecule, method=method
                )[0]
                ref_mol = rj.get_molecule()

                align_mols_map = final_mol.align(ref_mol, atoms_map=True)
                rmsd_val_map = align_mols_map[1]["rmsd"]
                rmsd_val_mirror = 1.0

                if rmsd_symm:
                    align_mols_mirror = final_mol.align(ref_mol, run_mirror=True)
                    rmsd_val_mirror = align_mols_mirror[1]["rmsd"]
                    out_string += """ 
        
        Comparing molecule {}  with molecule {} the RMSD values is: {} (with mirror {})
        
    """.format(
                        n, j[0], rmsd_val_map, rmsd_val_mirror
                    )
                out_string += """ 
        
        Comparing molecule {}  with molecule {} the RMSD values is: {}
        
    """.format(
                    n, j[0], rmsd_val_map
                )
                if rmsd_val_map < rmsd_val or rmsd_val_mirror < rmsd_val:
                    unique.append(False)
                    out_string += "OOPS molecule {} already exists. \n".format(j[0])
                else:
                    unique.append(True)
                    out_string += "Molecule {} does not exist yet\n".format(j[0])
            if all(unique):
                out_string += """
    
    Molecule {} was found to be unique, proceeding to opt_dset_name addition...
    
    """.format(
                    n
                )
                try:
                    ds_opt.add_entry(n, final_mol, save=True)
                    new += 1
                except KeyError as e:
                    print_out(str(e) + "\n")
            else:
                out_string += """
    
    Molecule {} already exists, discarting....
    
    """.format(
                    n
                )
            print_out(out_string)
        ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
        tot_mol = ""
        for i in ds_opt.df.index:
            tot_mol += i + " "
        out_string = """
    
    Number of structures processed: {}
    Total Number of molecules in the opt_dset_name: {}
    New molecules in round {}: {}
    Names of structures thus far: {}
    
    ****************************************End of round***************************************
    
    """.format(
            len(complete_opt_name), len(ds_opt.df.index), c, new, tot_mol
        )
        print_out(out_string)
        if single_site:
            break
        if new <= 1:
            print_out(
                "Found 1 or less new binding sites, so convergence will be declared.\n"
            )
            break
        c += 1
    print_out(
        "Finished sampling the cluster, found a total of  {}  binding sites.".format(
            len(ds_opt.df.index)
        )
    )

    # if max_rounds:
    #    if c == max_rounds:
    #        converged = True
    #        print_out(
    #            "Reached the maximum number of {}  binding sites searching rounds. Exiting...".format(
    #                max_rounds
    #            )
    #        )
    #        return converged

    # c += 1

    # if len(ds_opt.df.index) <= 16:
    #    continue

    # if (len(ds_opt.df.index) >= max_struct) or (c >= 7) or (new <= 1):
    #    converged = True
    #    print_out(
    #        "All or at least {} binding sites were found! Exiting...".format(
    #            max_struct
    #        )
    #    )
    #    return converged
