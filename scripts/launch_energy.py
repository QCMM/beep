import sys, time
#sys.path.append('/home/svogt/fractal/site_finder')#from molecule_sampler import molecule_sampler as  mol_sample
import qcfractal.interface as ptl
from pathlib import Path
from beep.binding_energy_compute import compute_be

from optparse import OptionParser
parser = OptionParser()


parser.add_option("-m",
                  "--molecule",
                  dest="s_mol",
                  help="The name of the molecule to be sampled (from small_mol collection)")
parser.add_option("-c",
                  "--collection",
                   dest="coll",
                   help="The name of the collection with the water clusters (default: Water_22)",
                   default="Water_22"
                   )

parser.add_option("-s",
                  "--small_collection",
                   dest="s_coll",
                   help="The name of the collection with molecules or radicals (default: Small_molecules)",
                   default="Small_molecules"
                   )

parser.add_option("-t",
                  "--level_of_theory_opt",
                  dest="ltheory_opt",
                  help=
                  "The level of theory of the optimization in the format: method_basis (default: hf3c_minix)",
                  default="hf3c_minix"
                 )
parser.add_option("-l",
                  "--level_of_theory",
                  dest="ltheory",
                  help=
                  "The level of theory in the format: method_basis (default: wpbe-d3bj_def2-tzvp)",
                  default="wpbe-d3bj_def2-tzvp"
                 )

parser.add_option("-k",
                  "--keyword_id",
                  dest="kw",
                  help="ID of the QC keywords (default: None)",
                  default=None)

options = parser.parse_args()[0]

method  = options.ltheory.split("_")[0]
basis  = options.ltheory.split("_")[1]
#program = options.program
opt_lot = options.ltheory_opt
lot = options.ltheory
kw_id   = options.kw
wat_collection = options.coll
small_collection = options.s_coll
smol_name = options.s_mol
size = wat_collection.split("_")[1]


frequency =  600

client = ptl.FractalClient(address="localhost:7777", verify=False, username="svogt", password="7kyRT-Mrow3jH0Lg6b9YIhEjAcvU9EpFBb9ouMClU5g")


#w_cluster_list = ["W22_01","W22_02","W22_03","W22_04","W22_05","W22_06","W22_09","W22_10","W22_11","W22_12"]
out_file = Path("/home/astrochem/energy_compute/"+str(smol_name)+"_en/W"+str(size)+"/out_sampl_"+str(options.ltheory_opt.split("_")[0])+".dat")

def print_out(string):
    with open(out_file, 'a') as f:
        f.write(string)

if not out_file.is_file():
    out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, 'w') as f:
    f.write('''        Welcome to the Energizer! 

Description: 

Author: svogt, gbovolenta
Data: 21/07/2021
version: 0.1.1

'''
    )
w_dict = {}
#for w in w_cluster_list:

w_cluster_list = []
for w in range(1,18):
    frame = "W"+str(size)+"_"+"%02d" %w
    if str(size) == "22":
        if w ==7 or w==8:
            continue
    w_cluster_list.append(frame)
    w_dict[frame] = False

#w_dict["W22_01"] = False
#print(w_dict.values())
#print(all(w_dict.values()))
#sys.exit(1)
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
            compute_be(wat_collection, small_collection, database, opt_lot, lot, out_file, client=client)

        #with open(out_file, 'a') as f:
        #    f.write('''
        #       Sending {} optimizations at {} level.'''.format(c,lot))

    time.sleep(frequency)

print_out("All the energies for species {} have been sent!\n".format(str(smol_name)))
                                                                                                                                                                                          140,1         Bot

