import sys, time
#sys.path.append('/home/svogt/fractal/site_finder')
#from molecule_sampler import molecule_sampler as  mol_sample
import qcfractal.interface as ptl
#import numpy as np
from pathlib import Path
#from converge_sampling import sampling
from beep.converge_sampling import sampling

from optparse import OptionParser
usage ='''python %prog [options]

A command line interface to sample the surface of a set of water clusters (stored in a 
QCFractal DataSet)  with a small molecule or atom. This CLI is part
of the Binding Energy Evaluation Platform (BEEP).
'''
parser = OptionParser(usage=usage)
parser.add_option("--client_address",
                  dest="client_address",
                  help="The URL address and port of the QCFractal server (default: localhost:7777)",
                  default="localhost:7777"
)
parser.add_option("--molecule",
                  dest="molecule",
                  help="The name of the molecule to be sampled (from a QCFractal OptimizationDataSet collection)"
)
parser.add_option("--surface_model_collection",
                   dest="surface_model_collection",
                   help="The name of the collection with the set of water clusters (dafault: ASW_22)",
                   default="ASW_12"
)
parser.add_option("--small_molecule_collection",
                   dest="small_molecule_collection",
                   help="The name of the collection containing molecules or radicals (dafault: Small_molecules)",
                   default="Small_molecules"
)
parser.add_option("--molecules_per_round",
                  dest="molecules_per_round",
                  type = "int",
                  help="Number of molecules to be optimized each round (Default = 10)",
                  default=10
)
parser.add_option("--sampling_shell",
                  dest="sampling_shell",
                  type = "float",
                  default=1.5,
                  help="The shell size of sampling space (Default = 1.5 Angstrom)"
)
parser.add_option("--maximal_binding_sites",
                  dest="maximal_binding_sites",
                  default= 21,
                  help="The maximal number of binding sites per cluster (default: 21)"
)
parser.add_option("-l",
                  "--level_of_theory",
                  dest="level_of_theory",
                  help=
                  "The level of theory in the format: method_basis (default: blyp_def2-svp)",
                  default="blyp_def2-svp"
                 )
parser.add_option("--refinement_level_of_theory",
                  dest="r_level_of_theory",
                  help=
                  "The level of theory for geometry refinement in the format: method_basis (default: hf3c_minix)",
                  default="hf3c_minix"
                 )
parser.add_option("--rmsd_value",
                  dest="rmsd_val",
                  default= 0.40,
                  help="Rmsd geometrical criteria, all structure below this value will not be considered as unique. (default: 0.40 angstrom)",
)
parser.add_option("--rmsd_symmetry",
                  action='store_true',
                  dest="rmsd_symmetry",
                  help="Consider the molecular symmetry for the rmsd calculation"
                        )
parser.add_option("-p",
                  "--program",
                  dest="program",
                  default="psi4",
                  help="The program to use for this calculation (default: psi4)",
                  )

parser.add_option("-k",
                  "--keyword_id",
                  dest="keyword_id",
                  help="ID of the QC keywords for the OptimizationDataSet specification (default: None)",
                  default=None)
parser.add_option("--print_out",
                  action='store_true',
                  dest="print_out",
                  help="Print an output"
)


def print_out(string, o_file):
    with open(o_file, 'a') as f:
        f.write(string)

options = parser.parse_args()[0]

method  = options.level_of_theory.split("_")[0]
basis  = options.level_of_theory.split("_")[1]
program = options.program
opt_lot = options.level_of_theory
r_lot = options.r_level_of_theory
kw_id   = options.keyword_id
s_shell = options.sampling_shell
num_struct = options.molecules_per_round
rmsd_symm = options.rmsd_symmetry
wat_collection = options.surface_model_collection
mol_collection = options.small_molecule_collection
smol_name = options.molecule
rmsd_val = options.rmsd_val
max_struct = options.maximal_binding_sites

client = ptl.FractalClient(address=options.client_address, verify=False)

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
    
    cluster = wat_collection+"_"+"%02d" %w
    opt_dset_name = str(smol_name)+ "_"+cluster
    try:
        ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
        c = len(ds_opt.status(collapse=False))
        count = count + int(c)
        continue
    except KeyError:
        out_file = Path("./site_finder/"+str(smol_name)+"_w/"+ cluster + "/out_sampl.dat")

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
    smpl_opt_dset_name = 'pre_'+str(opt_dset_name)
    s_conv = sampling(method, basis, program, opt_lot, kw_id, num_struct, rmsd_symm, wat_collection, cluster, mol_collection, smol_name, opt_dset_name, s_shell,  out_file, client)
    if s_conv:
       ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
       ds_opt.add_specification(**add_spec,overwrite=True)
       ds_opt.save()
       c=ds_opt.compute(m+"_"+b, tag=tag)
       count = count + int(c)
       with open(out_file, 'a') as f:
           f.write('''
           Sending {} optimizations at {} level.'''.format(c,r_lot))
    if count > 220: 
       break

                                            
