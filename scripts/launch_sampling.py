import sys, time
import qcfractal.interface as ptl
from pathlib import Path
from beep.converge_sampling import sampling
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
    A command line interface to sample the surface of a set of water clusters (stored in a 
    QCFractal DataSet) with a small molecule or atom. This CLI is part
    of the Binding Energy Evaluation Platform (BEEP).
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
        required=True,
        help="The name of the molecule to be sampled (from a QCFractal OptimizationDataSet collection)",
    )
    parser.add_argument(
        "--surface_model_collection",
        default="Water_22",
        help="The name of the collection with the set of water clusters (default: Water_22)",
    )
    parser.add_argument(
        "--small_molecule_collection",
        default="Small_molecules",
        help="The name of the collection containing molecules or radicals (default: Small_molecules)",
    )
    parser.add_argument(
        "--sampling_shell",
        type=float,
        default=2.0,
        help="The shell size of sampling space in Angstrom (Default = 2.0)",
    )
    parser.add_argument(
        "--sampling_condition",
        type=str,
        default="normal",
        help="How tight the sampling should be done for each surface. Options: sparse, normal, fine (Default: normal)",
    )
    parser.add_argument(
        "--level_of_theory",
        default="blyp_def2-svp",
        help="The level of theory in the format: method_basis (default: blyp_def2-svp)",
    )
    parser.add_argument(
        "--refinement_level_of_theory",
        default="hf3c_minix",
        help="The level of theory for geometry refinement in the format: method_basis (default: hf3c_minix)",
    )
    parser.add_argument(
        "--rmsd_value",
        type=float,
        default=0.40,
        help="Rmsd geometrical criteria, all structures below this value will not be considered as unique. (default: 0.40 angstrom)",
    )
    parser.add_argument(
        "--rmsd_symmetry",
        action="store_true",
        help="Consider the molecular symmetry for the rmsd calculation",
    )
    parser.add_argument(
        "--program",
        default="psi4",
        help="The program to use for this calculation (default: psi4)",
    )
    parser.add_argument(
        "--sampling_tag",
        default="sampling",
        help="The tag to use to specify the qcfractal-manager for the sampling optimization (default: sampling)",
    )
    parser.add_argument(
        "--keyword_id",
        default=None,
        help="ID of the QC keywords for the OptimizationDataSet specification (default: None)",
    )

    return parser.parse_args()


options = parse_arguments()

username = options.usern
password = options.passwd
method = options.level_of_theory.split("_")[0]
basis = options.level_of_theory.split("_")[1]
program = options.program
tag = options.tag
opt_lot = options.level_of_theory
r_lot = options.r_level_of_theory
kw_id = options.keyword_id
sampling_shell = options.sampling_shell
sampling_condition = options.sampling_condition
rmsd_symm = options.rmsd_symmetry
cluster_collection = options.surface_model_collection
mol_collection = options.small_molecule_collection
smol_name = options.molecule
rmsd_val = options.rmsd_val

client = ptl.FractalClient(
    address=options.client_address, verify=False, username=username, password=password
)

m = r_lot.split("_")[0]
b = r_lot.split("_")[1]


spec = {
    "name": m + "_" + b,
    "description": "Geometric + Psi4/" + m + "/" + b,
    "optimization_spec": {"program": "geometric", "keywords": None},
    "qc_spec": {
        "driver": "gradient",
        "method": m,
        "basis": b,
        "keywords": None,
        "program": "psi4",
    },
}

count = 0

try:
    ds_soc = client.get_collection("OptimizationDataset", cluster_collection)
except KeyError:
    print(
        """Collection with set of clusters that span the surface {} does not exist. Please create it first. Exiting...
    """.format(
            cluster_collection
        )
    )
    sys.exit(1)

try:
    ds_sm = client.get_collection("OptimizationDataset", mol_collection)
except KeyError:
    print(
        """Collection {} with the target molecules does not exist, please create it first. Exiting...
    """.format(
            mol_collection
        )
    )
    sys.exit(1)

soc_list = ds_soc.data.records

try:
    target_mol = ds_sm.get_record(smol_name, opt_lot).get_final_molecule()
except KeyError:
    print(
        "{} is not optimized at the requested level of theory, please optimize them first\n".format(
            smol_name
        )
    )
    sys.exit(1)

for w in ds_soc.data.records:
    print("Processing cluster: {}".format(w))
    try:
        cluster = ds_soc.get_record(w, opt_lot).get_final_molecule()
    except KeyError:
        print(
            "{} is not optimized at the requested level of theory, please optimize it first\n".format(
                w
            )
        )
        continue

    opt_dset_name = smol_name + "_" + w

    try:
        ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
        c = len(ds_opt.status(collapse=False))
        count = count + int(c)
    except KeyError:
        pass

    out_file = Path("./site_finder/" + str(smol_name) + "_w/" + w + "/out_sampl.dat")

    if not out_file.is_file():
        out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        f.write(
            """        Welcome to the Molecular Sampler Converger! 
    
    Description: The molecular sampler converger optimizes random initial configurations
    of a small molecule around a  center molecule until at least 220 binding sites are found or
    all clusters of a set are sampled..
    
    Author: svogt, gbovolenta
    Data: 10/24/2020
    
    """
        )
    sampling(
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
        out_file,
        client,
        sampling_shell,
        sampling_condition,
    )
    print("Total number of binding sites so far: {} ".format(count))
    ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
    ds_opt.add_specification(**spec, overwrite=True)
    ds_opt.save()
    c = ds_opt.compute(m + "_" + b, tag="refinement")
    count = count + int(c)
    with open(out_file, "a") as f:
        f.write(
            """
        Sending {} optimizations at {} level.""".format(
                c, r_lot
            )
        )
    if count > 220:
        break
