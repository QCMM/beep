import sys, time
import qcfractal.interface as ptl
from pathlib import Path
from beep.binding_energy_compute import compute_be, compute_hessian

from optparse import OptionParser
usage='''python %prog [options]

A command line interface to compute the binding energy of a set of optimized binding sites
of a set of water clusters (stored in a 
QCFractal DataSet). This CLI is part of the Binding Energy Evaluation Platform (BEEP).
'''
parser = OptionParser(usage=usage)

parser.add_option("--client_address",
                  dest="client_address",
                  help="The URL address and port of the QCFractal server (default: localhost:7777)",
                  default="localhost:7777"
)
parser.add_option("--username",
                  dest="usern",
                  help="The username for the database client (Default = None)",
                  default=None
)
parser.add_option("--password",
                  dest="passwd",
                  help="The password for the database client (Default = None)",
                  default=None
)
parser.add_option("--surface-model-collection",
                   dest="surface-model-collection",
                   help="The name of the collection with the water clusters (default: Water_22)",
                   default="Water_22"
                   )

parser.add_option("--small_molecule_collection",
                   dest="small_molecule_collection",
                   help="The name of the collection with small molecules or radicals (default: Small_molecules)",
                   default="Small_molecules"
                   )
parser.add_option("--molecule",
                  dest="molecule",
                  help="The name of the molecule for the binding energy computation"
                  )
parser.add_option("--level_of_theory",
                  dest="level_of_theory",
                  help="The level of theory for the binding energy computation in the format: method_basis (default: wpbe-d3bj_def2-tzvp)",
                  default="wpbe-d3bj_def2-tzvp"
                 )
parser.add_option("--opt_level_of_theory",
                  dest="opt_lot",
                  help=
                  "The level of theory  of the binding sites optimization in the format: method_basis (default: hf3c_minix)",
                  default="hf3c_minix"
                 )

parser.add_option("--keyword_id",
                  dest="keyword_id",
                  help="ID of the QCfractal for the single point computations keywords (default: None)",
                  default=None)

parser.add_option("--hessian_compute",
                  dest="hessian",
                  help="Computes the hessian for the molecules comprising the binding sites of model cluster X. If 0 is specified, no Hessian is computed (default = 1)",
                  default=1)

parser.add_option("-p",
                  "--program",
                  dest="program",
                  default="psi4",
                  help="The program to use for this calculation (default: psi4)",
                  )

parser.add_option("--energy_tag",
                  dest="energy_tag",
                  default="energies",
                  help="The tag to used to specify the qcfractal-manager for the BE computations  (default: energies)",
parser.add_option("--hessian_tag",
                  dest="hess_tag",
                  default="hessian",
                  help="The tag to used to specify the qcfractal-manager for the hessian  (default: hessian)",
                 )
options = parser.parse_args()[0]

username = options.usern
password = options.passwd
wat_collection = options.surface_model_collection
mol_collection = options.small_molecule_collection
program = options.program
opt_lot = options.opt_lot
lot = options.level_of_theory
kw_id   = options.keyword_id
smol_name = options.molecule
hessian = options.hessian
be_tag= options.energy_tag
hess_tag= options.hess_tag

frequency =  600


client = ptl.FractalClient(address=options.client_address, verify=False, username = username, password=password)

try:
    ds_sm = client.get_collection("OptimizationDataset", mol_collection)
except KeyError:
    print("""Collection {} with the target molecules does not exist, please create it first. Exiting...
    """.format(mol_collection))
    sys.exit(1)

try:
    target_mol = ds_sm.get_record(smol_name, opt_lot).get_final_molecule()
except KeyError:
    print("{} is not optimized at the requested level of theory, please optimize them first\n".format(smol_name))
    sys.exit(1)

try:
    ds_soc = client.get_collection("OptimizationDataset", wat_collection)
except KeyError:
    print("""Collection with set of clusters that span the surface {} does not exist. Please create it first. Exiting...
    """.format(wat_collection))
    sys.exit(1)


out_file = Path(str(smol_name)+"_en/"+ds_soc.name+"/out_en_"+str(opt_lot.split("_")[0])+".dat")

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
        print("{} is not optimized at the requested level of theory, please optimize it first\n".format(w))
        sys.exit(1)
    w_cluster_list.append(w)
    w_dict[w] = False

while not all(w_dict.values()):
    for w in w_cluster_list:
        print("Processing cluster: {}".format(w))
        if w_dict[w] == True:
           continue

        database = str(smol_name)+"_"+w
        try:
            ds_opt = client.get_collection("OptimizationDataset", database)
        except:
            "KeyError"
            print_out("Optimization database {} does not exist\n".format(str(database)))
            continue

        try:
            ds_opt.query(opt_lot)
        except:
            "KeyError"
            print_out("Optimization of {}  has not proceeded at the {} level of theory\n".format(str(database), opt_lot))
            continue
        

        if not ds_opt.status(status='INCOMPLETE', specs= opt_lot).empty:
            print_out("Collection {}: Some optimizations are still running\n".format(str(database)))
            w_dict[w] = False
        else:
            print_out("\n\nCollection {}: All optimizations finished!\n".format(str(database)))
            w_dict[w] = True
            print_out("Time to send the energies!\n")
            compute_be(wat_collection, mol_collection, database, opt_lot, lot, out_file, energy_tag, client=client, program=program)
            time.sleep(30)

        if float(w.split('_')[1]) == float(hessian):
           name_be = "be_" + str(database) + "_" + opt_lot.split("_")[0]
           compute_hessian(name_be, opt_lot, out_file, hess_tag, client=client, program=program)
           print_out("Sending Hessian computation for cluster {}\n".format(w))

    time.sleep(frequency)

print_out("All the energies (and hessians) for species {} have been sent!\n".format(str(smol_name)))
