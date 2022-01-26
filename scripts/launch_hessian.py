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
                   help="The name of the collection with the water clusters (default: Small_molecules)",
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
                  default=None
                  )
parser.add_option("-n",
                  "--number_of_collections",
                  dest="numcoll",
                  help="The number of collections that will be computed (1 collection ~ 25 structures, default: 1)",
                  )
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
num_coll = options.numcoll

frequency =  600

client = ptl.FractalClient(address="localhost:7777", verify=False, username="svogt", password="7kyRT-Mrow3jH0Lg6b9YIhEjAcvU9EpFBb9ouMClU5g")


out_file = Path("./energy_compute/"+str(smol_name)+"_hess/W"+str(size)+"/out_sampl_"+str(options.ltheory_opt.split("_")[0])+".dat")

def print_out(string):
    with open(out_file, 'a') as f:
        f.write(string)

if not out_file.is_file():
    out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, 'w') as f:
    f.write('''        Welcome to the Hessian Calculation! 

Description: 

Author: svogt, gbovolenta
Data: 21/07/2021
version: 0.1.1

'''
    )
#w_dict = {}
#for w in w_cluster_list:

w_cluster_list = []
for w in range(1, num_coll+1):
    frame = "W"+str(size)+"_"+"%02d" %w
    if str(size) == "22":
        if w ==7 or w==8:
            continue
    w_cluster_list.append(frame)
for w in w_cluster_list:

    #if w_dict[w] == True:
 #      continue 
#
    #database = str(smol_name)+"_"+w
    database = "be_"+str(smol_name)+"_"+frame+"_"+opt_lot.split("_")[0]
    print_out("Reaction  database: {} exists\n".format(str(database)))
    try:
        ds_be = client.get_collection("ReactionDataset", database)
    except:
        "KeyError"
        print_out("Reaction  database {} does not exist\n".format(str(database)))
        continue


    ds_be.get_values()
    print(ds_be.get_values())
    #print(ds_be.get_records('WPBE',stoich= "be_nocp"))
    mols = []



    for i in ds_be.df.index:
        #print_out("Molecule {}\n".format(i))
        for j in range(3):
            #print(j)
            if j == 2:
                j=3
            #if method == 'hf3c':
            m_be = (str(method).split("-")[0]).upper()
            #m_be = 'WPBE'
            #print_out("One time please:  {}".format(m_be))
            rr = ds_be.get_records(method = m_be, stoich= "be_nocp").loc[i]['record'][j]
            #else:
            #    rr = ds_be.get_records(method,stoich= "default").loc[i]['record'][j]
            #print(rr)
            m = rr.get_molecule()
            if m in mols:

                continue
            mols.append(m)
    
    if small_collection == "Small_radicals":
        kw = ptl.models.KeywordSet(**{"values": {'function_kwargs': {'dertype': 1},'reference': 'uhf'}})
    else:
        kw = ptl.models.KeywordSet(**{"values": {'function_kwargs': {'dertype': 1}}})
    kw_id = client.add_keywords([kw])[0]
    r = client.add_compute("psi4", opt_lot.split("_")[0], opt_lot.split("_")[1], "hessian", kw_id, mols,tag='hessian_comp')
    print_out("{} hessian computations have been sent.\n".format(r))

print_out("All the hessians for species {} have been sent!\n".format(str(smol_name)))
                                        
