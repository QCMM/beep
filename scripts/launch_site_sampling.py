import sys, time
import qcfractal.interface as ptl
from pathlib import Path
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
parser.add_option("--cluster-name",
                  dest="cluster_name",
                  help="The QCFractal name of the cluster to be sampled, as it appears in the OptimizationDataset (e.g co_W22_02_0009)",
)
parser.add_option("--sampling-molecule-id",
                   dest="sampling_mol_id",
                   help="The QCFractal ID of the molecule that samples the cluster",
)
parser.add_option("--molecule-size",
                  dest="molecule_size",
                  help="The size of the molecule bound to the cluster to be sampled"
)
parser.add_option("--sampling_shell",
                  dest="sampling_shell",
                  type = "float",
                  default=2.5,
                  help="Distance where the sampling molecule should be set (Default = 2.5 angstrom) "
)
parser.add_option("--zenith-angle",
                  dest="zenith_angle",
                  type = "float",
                  default=3.14159/2,
                  help="The angle with respect to the zenith to construct the sampling shell (Default: pi/2)",
)
parser.add_option("--maximal_binding_sites",
                  dest="maximal_binding_sites",
                  type = "int",
                  default= 15,
                  help="The maximal number of binding sites per cluster (default: 21)"
)
parser.add_option("--number-of-rounds",
                  dest="number_of_rounds",
                  type = "int",
                  default= 1,
                  help="The maximal number of rounds a cluster should be sampled (default: 1)"
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
                  type = "float",
                  default= 0.10,
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
parser.add_option("--sampling_tag",
                  dest="tag",
                  default="sampling",
                  help="The tag to used to specify the qcfractal-manager for the sampling optimization  (default: sampling)",
                  )

parser.add_option("-k",
                  "--keyword_id",
                  dest="keyword_id",
                  help="ID of the QC keywords for the OptimizationDataSet specification (default: None)",
                  default=None)
#parser.add_option("--print_out",
#                  action='store_true',
#                  dest="print_out",
#                  help="Print an output"
#)


def print_out(string, o_file):
    with open(o_file, 'a') as f:
        f.write(string)

options = parser.parse_args()[0]

username = options.usern
password = options.passwd
method  = options.level_of_theory.split("_")[0]
basis  = options.level_of_theory.split("_")[1]
program = options.program
tag = options.tag
kw_id   = options.keyword_id
opt_lot = options.level_of_theory
rmsd_symm = options.rmsd_symmetry
rmsd_val = options.rmsd_val

cluster_name = options.cluster_name
sampling_mol_id = options.sampling_mol_id
sampled_mol_size = options.molecule_size
molecule_size = options.molecule_size
s_shell = options.sampling_shell
zenith_angle = options.zenith_angle
max_struct = options.maximal_binding_sites
max_rounds = options.number_of_rounds
noise = options.noise
purge = options.purge

#grid_size = options.grid_size


r_lot = options.r_level_of_theory
single_site = True

client = ptl.FractalClient(address=options.client_address, verify=False, username = username, password=password)

try:
    ds_opt = client.get_collection("OptimizationDataset", "_".join(cluster_name.split('_')[:-1]))
except KeyError:
    print("""Collection with structure to be sampled  {} does not exist. Please check name. Exiting...
    """.format(wat_collection))
    sys.exit(1)

try:
    cluster = ds_opt.get_record(cluster_name, opt_lot).get_final_molecule()
except KeyError:
    print("{} is not optimized at the requested level of theory or does not exist. \n".format(cluster_name))
    sys.exit(1)


m = r_lot.split('_')[0]
b = r_lot.split('_')[1]


add_spec = {'name': m+'_'+b,
        'description': 'Geometric + Psi4/'+m+'/'+b,
        'optimization_spec': {'program': 'geometric', 'keywords': None},
        'qc_spec': {'driver': 'gradient',
        'method': m,
        'basis': b,
        'keywords': None,
        'program': 'psi4'}}

count = 0

sampling_mol =  client.query_molecules(int(sampling_mol_id))[0]

if not (sampling_mol):
    print("Target molecule does not exist, check your id\n")
    sys.exit(1)

print("Processing cluster: {}".format(cluster_name))

opt_dset_name = cluster_name+"+"+sampling_mol.name

try:
    ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
    c = len(ds_opt.status(collapse=False))
    count = count + int(c)
except KeyError:
    pass

out_file = Path("./site_finder/"+str(sampling_mol.name)+"_w/"+ cluster_name + "/out_sampl.dat")

if not out_file.is_file():
    out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, 'w') as f:
    f.write('''        Welcome to the Single site sampler ! 

Description: The molecular sampler converger optimizes configurations 
of a small molecule around a  binding site target  molecule until no more unique stationary
points are found.

Author: svogt, gbovolenta
Data: 10/24/2020
version: 0.1.2

'''
    )
s_conv = sampling(method, basis, program, tag, kw_id, opt_dset_name, opt_lot, rmsd_symm,
                  rmsd_val, sampling_mol, cluster, out_file, client,
                  max_struct=max_struct, sampled_mol_size = sampled_mol_size, 
                  zenith_angle=zenith_angle, max_rounds=max_rounds,
                  single_site=single_site, sampling_shell = s_shell)

print("Total number of binding sites so far: {} ".format(count))
if s_conv:
   ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
   ds_opt.add_specification(**add_spec,overwrite=True)
   ds_opt.save()
   c=ds_opt.compute(m+"_"+b, tag="refinement")
   count = count + int(c)
   with open(out_file, 'a') as f:
       f.write('''
       Sending {} optimizations at {} level.'''.format(c,r_lot))
else: 
    print("An error occured. Check the output in the site-finder folder")
    sys.exit(1)
