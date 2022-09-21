import sys, time
from .molecule_sampler import molecule_sampler as mol_sample
import qcfractal.interface as ptl
import numpy as np
from pathlib import Path
from optparse import OptionParser


def sampling(
    method,
    basis,
    program,
    opt_lot,
    tag,
    kw_id,
    num_struct,
    max_struct,
    rmsd_symm,
    rmsd_val,
    target_mol,
    wat_cluster,
    opt_dset_name,
    sampling_shell,
    o_file,
    client
):
    def print_out(string):
        with open(o_file, "a") as f:
            f.write(string)

    smpl_opt_dset_name = "pre_" + str(opt_dset_name)


    print_out('''
    Water cluster: {}
    Small molecule: {}
    method: {}
    basis:  {}
    '''.format(wat_cluster, target_mol, method, basis)   
    )

    frequency = 600

    out_string = ""

    try:
        smpl_ds_opt = client.get_collection("OptimizationDataset", smpl_opt_dset_name)
        out_string += """OptimizationDateset {} already exists, new sampled structures will be saved here.
        """.format(
            smpl_opt_dset_name
        )
    except KeyError:
        smpl_ds_opt = ptl.collections.OptimizationDataset(smpl_opt_dset_name, client=client)
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
        "optimization_spec": {"program": "geometric", "keywords": None},
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
    Maximum number of structures to look for:  {}
    
    Starting Convergence procedure....
    
    """.format(
        program, method, basis, max_struct
    )

    print_out(out_string)

    c = 1
    converged = False
    nmol = 0
    while not converged:
        out_string = ""
        mol_l = " "
        entry_list = []
        complete_opt_name = []

        molecules = mol_sample(
            wat_cluster,
            target_mol,
            number_of_structures=num_struct,
            sampling_shell=sampling_shell,
            print_out=False,
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
                c, num_struct, mol_l
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
        # if sampl_only:
        #    print_out(str(num_struct)+" molecules where sampled and are optimizing. Thank you for using molecule sampler!")
        #    break
        print_out("Waiting for optimizations to complete\n\n")

        # Checks if no more jobs are running

        status = []

        while not jobs_complete:
            for i in pid.split():
                rr = client.query_procedures(int(i))[0]
                status.append(rr.status)

            # Initaial spec Query to avoid status bug
            smpl_ds_opt = client.get_collection("OptimizationDataset", smpl_opt_dset_name)
            smpl_ds_opt.query(opt_lot)

            #if not smpl_ds_opt.status(status="INCOMPLETE", specs=opt_lot).empty:
            #    print_out("Some jobs are still running, will sleep now\n")
            #    time.sleep(frequency)
            if "INCOMPLETE" in status:
                print_out("Some jobs are still running, will sleep now\n")
                time.sleep(frequency)

            else:
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
            # Comparing molecules to already existing molecules in the Optopt_dset_name, only adding them if RMSD > 0.25
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
        c += 1
        if len(ds_opt.df.index) <= 13:
            continue
        if (len(ds_opt.df.index) >= max_struct) or c >= 7 or (new <= 1):
            converged = True
            print_out(
                "All or at least {} binding sites were found! Exiting...".format(
                    max_struct
                )
            )
            return converged
