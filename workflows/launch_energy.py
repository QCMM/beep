import sys
import time
import argparse
import qcfractal.interface as ptl
from pathlib import Path
from beep.binding_energy_compute import compute_be, compute_hessian

usage = '''python %(prog)s [options]

A command line interface to compute the binding energy of a set of optimized binding sites
of a set of water clusters (stored in a QCFractal DataSet). This CLI is part of the Binding Energy Evaluation Platform (BEEP).
'''

parser = argparse.ArgumentParser(description=usage)

parser.add_argument("--client_address",
                    help="The URL address and port of the QCFractal server (default: localhost:7777)",
                    default="localhost:7777"
)
parser.add_argument("--username",
                    help="The username for the database client (Default = None)",
                    default=None
)
parser.add_argument("--password",
                    help="The password for the database client (Default = None)",
                    default=None
)
parser.add_argument("--surface_model_collection",
                    help="The name of the collection with the water clusters (default: Water_22)",
                    default="Water_22"
)
parser.add_argument("--small_molecule_collection",
                    help="The name of the collection with small molecules or radicals (default: Small_molecules)",
                    default="Small_molecules"
)
parser.add_argument("--molecule",
                    help="The name of the molecule for the binding energy computation"
)
parser.add_argument("--level_of_theory",
                    nargs='+',
                    help="The level(s) of theory for the binding energy computation in the format: method_basis (default: wpbe-d3bj_def2-tzvp)",
                    default=["wpbe-d3bj_def2-tzvp"]
)
parser.add_argument("--opt_level_of_theory",
                    help="The level of theory of the binding sites optimization in the format: method_basis (default: hf3c_minix)",
                    default="hf3c_minix"
)
parser.add_argument("--keyword_id",
                    help="ID of the QCfractal for the single point computations keywords (default: None)",
                    default=None
)
parser.add_argument("--hessian_compute",
                    default=None,
                    help="Computes the hessian for the molecules comprising the binding sites of the chosen cluster. If None is specified, no Hessian is computed (default = None)",
)
parser.add_argument("-p", "--program",
                    default="psi4",
                    help="The program to use for this calculation (default: psi4)"
)
parser.add_argument("--energy_tag",
                    default="energies",
                    help="The tag to use to specify the qcfractal-manager for the BE computations (default: energies)"
)
parser.add_argument("--hessian_tag",
                    default="hessian",
                    help="The tag to use to specify the qcfractal-manager for the hessian (default: hessian)"
)

args = parser.parse_args()

username = args.username
password = args.password
wat_collection = args.surface_model_collection
mol_collection = args.small_molecule_collection
program = args.program
opt_lot = args.opt_level_of_theory
lot_list = args.level_of_theory
kw_id = args.keyword_id
smol_name = args.molecule
hessian_clust = args.hessian_compute
energy_tag = args.energy_tag
hess_tag = args.hessian_tag

frequency = 600

client = ptl.FractalClient(address=args.client_address, verify=False, username=username, password=password)

try:
    ds_sm = client.get_collection("OptimizationDataset", mol_collection)
except KeyError:
    print(f"Collection {mol_collection} with the target molecules does not exist, please create it first. Exiting...")
    sys.exit(1)

try:
    target_mol = ds_sm.get_record(smol_name, opt_lot).get_final_molecule()
except KeyError:
    print(f"{smol_name} is not optimized at the requested level of theory, please optimize them first\n")
    sys.exit(1)

try:
    ds_soc = client.get_collection("OptimizationDataset", wat_collection)
except KeyError:
    print(f"Collection with set of clusters that span the surface {wat_collection} does not exist. Please create it first. Exiting...")
    sys.exit(1)

out_file = Path(f"{smol_name}_en/{ds_soc.name}/out_en_{opt_lot.split('_')[0]}.dat")

def print_out(string):
    with open(out_file, 'a') as f:
        f.write(string)

if not out_file.is_file():
    out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, 'w') as f:
    f.write('''        Welcome to the BEnergizer! 

Description: 

Author: svogt, gbovolenta
Data: 21/07/2021
version: 0.2.1

'''
    )

w_dict = {}
w_cluster_list = []

for w in ds_soc.data.records:
    try:
        wat_cluster = ds_soc.get_record(w, opt_lot).get_final_molecule()
    except KeyError:
        print(f"{w} is not optimized at the requested level of theory, please optimize it first\n")
        sys.exit(1)
    w_cluster_list.append(w)
    w_dict[w] = False

while not all(w_dict.values()):
    for w in w_cluster_list:
        print(f"Processing cluster: {w}")
        if w_dict[w]:
            continue

        database = f"{smol_name}_{w}"
        try:
            ds_opt = client.get_collection("OptimizationDataset", database)
        except KeyError:
            print_out(f"Optimization database {database} does not exist\n")
            continue

        try:
            ds_opt.query(opt_lot)
        except KeyError:
            print_out(f"Optimization of {database} has not proceeded at the {opt_lot} level of theory\n")
            continue

        if not ds_opt.status(status='INCOMPLETE', specs=opt_lot).empty:
            print_out(f"Collection {database}: Some optimizations are still running\n")
            w_dict[w] = False
        else:
            print_out(f"\n\nCollection {database}: All optimizations finished!\n")
            w_dict[w] = True
            print_out("Time to send the energies!\n")
            compute_be(wat_collection, mol_collection, database, opt_lot, lot_list, out_file, energy_tag, client=client, program=program)
            time.sleep(5)

        if hessian_clust is not None and w == hessian_clust:
            print("Sending Hessian computation for cluster {w}\n")
            name_be = f"be_{database}_{opt_lot.split('_')[0]}"
            print_out(f"Sending Hessian computation for cluster {w}\n")
            compute_hessian(name_be, opt_lot, out_file, hess_tag, client=client, program=program)

    time.sleep(frequency)

print_out(f"All the energies (and hessians) for species {smol_name} have been sent!\n")

