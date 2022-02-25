import sys, time
#import qcfractal.interface as ptl
import qcportal as ptl
from pathlib import Path
#from beep.binding_energy_compute import compute_be

from optparse import OptionParser
usage='''python %prog [options]

A command line interface to compute the binding energy of a set of optimized binding sites
of a set of water clusters (stored in a 
QCFractal DataSet). This CLI is part of the Binding Energy Evaluation Platform (BEEP).
'''
parser = OptionParser(usage=usage)

parser.add_option("--cluster_collection",
                   dest="cluster_collection",
                   help="The name of the collection with the water clusters (default: Water_22)",
                   default="Water_22"
                   )

parser.add_option("--small_mol_collection",
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
                  "The level of theory of for the optimization of the binding sites in the format: method_basis (default: hf3c_minix)",
                  default="hf3c_minix"
                 )

parser.add_option("--keyword_id",
                  dest="keyword_id",
                  help="ID of the QCfractal for the single point computations keywords (default: None)",
                  default=None)
parser.add_option("--hessian_compute",
                  dest="hessian",
                  help="Computes the hessian for the molecules comprising the binding sites of model cluster X. If 0 is specified, no Hessian is computed (defualt = 1)",
                  default=1)

parser.add_option("-p",
                  "--program",
                  dest="program",
                  default="psi4",
                  help="The program to use for this calculation (default: psi4)",
                  )

options = parser.parse_args()[0]

wat_collection = options.cluster_collection
small_collection = options.small_molecule_collection
program = options.program
opt_lot = options.opt_lot
lot = options.level_of_theory
kw_id   = options.keyword_id
smol_name = options.molecule
hessian = options.hessian
size = wat_collection.split("_")[1]

sys.exit(1)
frequency =  600

client = ptl.FractalClient(address="localhost:7777", verify=False, username="", password="")


out_file = Path(str(smol_name)+"_en/W"+str(size)+"/out_sampl_"+str(opt_lot.split("_")[0])+".dat")

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


for w in range(1,20):
    frame = "W"+str(size)+"_"+"%02d" %w
    w_cluster_list.append(frame)
    w_dict[frame] = False

while not all(w_dict.values()):
    for w in w_cluster_list:

        if w_dict[w] == True:
           continue

        database = str(smol_name)+"_"+w
        try:
            ds_opt = client.get_collection("OptimizationDataset", database)
        except:
            "KeyError"
            print_out("Optimization database {} does not exist".format(str(database)))
            continue

        ds_opt.query(opt_lot)

        if not ds_opt.status(status='INCOMPLETE', specs= opt_lot).empty:
            print_out("Collection {}: Some optimizations are still running\n".format(str(database)))
            w_dict[w] = False
        else:
            print_out("\n\nCollection {}: All optimizations finished!\n".format(str(database)))
            w_dict[w] = True
            print_out("Time to send the energies!\n")
            compute_be(wat_collection, small_collection, database, opt_lot, lot, out_file, client=client, program=program)
            time.sleep(30)

        if float(w.split('_')[1]) == float(hessian):
           name_be = "be_" + str(database) + "_" + opt_lot.split("_")[0]
           compute_hessian(name_be, opt_lot, out_file, client=client, program=program)

    time.sleep(frequency)

print_out("All the energies (and hessians) for species {} have been sent!\n".format(str(smol_name)))
