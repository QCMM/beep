import sys, time
#sys.path.append('/home/svogt/fractal/site_finder')
#from molecule_sampler import molecule_sampler as  mol_sample
import qcfractal.interface as ptl
#import numpy as np
from pathlib import Path
#from converge_sampling import sampling
from beep.converge_sampling import sampling

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-m",
                  "--molecule",
                  dest="s_mol",
                  help="The name of the molecule to be sampled (from small_mol collection)")
parser.add_option("-c",
                  "--collection",
                   dest="coll",
                   help="The name of the collection with the water clusters (dafault: Water_22)",
                   default="Water_22"
)
parser.add_option("-s",
                  "--mol_collection",
                   dest="m_coll",
                   help="The name of the collection containing molecules or radicals (dafault: Small_molecules)",
                   default="Small_molecules"
)
parser.add_option("-n",
                  "--molecules_per_round",
                  dest="s_num",
                  type = "int",
                  help="Number of molecules to be optimized each round (Default = 10)",
                  default=10
)
parser.add_option("-e",
                  "--sampling_shell",
                  dest="sampling_shell",
                  type = "float",
                  default=1.5,
                  help="The shell size of sampling space (Default = 1.5 Angstrom)"
)
parser.add_option("-l",
                  "--level_of_theory",
                  dest="ltheory",
                  help=
                  "The level of theory in the format: method_basis (default: blyp_def2-svp)",
                  default="blyp_def2-svp"
                 )
parser.add_option("-r",
                  "--refinement_level_of_theory",
                  dest="r_ltheory",
                  help=
                  "The level of theory for geometry refinement in the format: method_basis (default: hf3c_minix)",
                  default="hf3c_minix"
                 )
parser.add_option("--print_out",
                  action='store_true',
                  dest="print_out",
                  help="Print an output"
                        )
parser.add_option("--rmsd_symmetry",
                  action='store_true',
                  dest="symm",
                  help="Consider the molecular symmetry for the rmsd calculation"
                        )
parser.add_option("-d",
                  "--rmsd_value",
                  dest="rmsd_val",
                  default= 0.40,
                  help="Rmsd geometrical criteria (default: 0.41 angstrom)",
)
parser.add_option("-t",
                  "--max_structures",
                  dest="max_structures",
                  default= 21,
                  help="Max number of binding sites per cluster (default: 21)"
)
parser.add_option("-p",
                  "--program",
                  dest="program",
                  default="psi4",
                  help="The program to use for this calculation (default: psi4)",
                  )

parser.add_option("-k",
                  "--keyword_id",
                  dest="kw",
                  help="ID of the QC keywords (default: None)",
                  default=None)


def print_out(string, o_file):
    with open(o_file, 'a') as f:
        f.write(string)


options = parser.parse_args()[0]

method  = options.ltheory.split("_")[0]
basis  = options.ltheory.split("_")[1]
program = options.program
opt_lot = options.ltheory
r_lot = options.r_ltheory
kw_id   = options.kw
s_shell = options.sampling_shell
num_struct = options.s_num
rmsd_symm = options.symm
wat_collection = options.coll
mol_collection = options.m_coll
smol_name = options.s_mol
rmsd_val = options.rmsd_val
max_struct = options.max_structures 

client = ptl.FractalClient(address="localhost:7777", verify=False, username="svogt", password="7kyRT-Mrow3jH0Lg6b9YIhEjAcvU9EpFBb9ouMClU5g")

m = r_lot.split('_')[0]
b = r_lot.split('_')[1]
tag = 'refinement'


add_spec = {'name': m+'_'+b,
        'description': 'Geometric + Psi4/'+m+'/'+b,
        'optimization_spec': {'program': 'geometric', 'keywords': None},
        'qc_spec': {'driver': 'gradient',
        'method': m,
        'basis': b,
        'keywords': None,
        'program': 'psi4'}}



count = 0

for w in range(1,15):
    
    frame = wat_collection+"_"+"%02d" %w
    if w ==7 or w==8:
        continue
    database = str(smol_name)+ "_"+frame
    try:
        ds_opt = client.get_collection("OptimizationDataset", database)
        c = len(ds_opt.status(collapse=False))
        count = count + int(c)
        continue
    except KeyError:
        out_file = Path("./site_finder/"+str(smol_name)+"_w/"+ frame + "/out_sampl.dat")

    if not out_file.is_file():
        out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, 'w') as f:
        f.write('''        Welcome to the Molecular Sampler Converger! 
    
    Description: The molecular sampler converger optimizes random initial configurations 
    of a small molecule around a  center molecule until no more unique stationary 
    points are found.
    
    Author: svogt, gbovolenta
    Data: 10/24/2020
    version: 0.1.2
    
    '''
        )
    smpl_database = 'pre_'+str(database)
    s_conv = sampling(method, basis, program, opt_lot, kw_id, num_struct, rmsd_symm, wat_collection, frame, smol_name, database, s_shell,  out_file, client=client)
    if s_conv:
       ds_opt = client.get_collection("OptimizationDataset", database)
       ds_opt.add_specification(**add_spec,overwrite=True)
       ds_opt.save()
       c=ds_opt.compute(m+"_"+b, tag=tag)
       count = count + int(c)
       with open(out_file, 'a') as f:
           f.write('''
           Sending {} optimizations at {} level.'''.format(c,r_lot))
    if count > 220: 
       break

                                            
