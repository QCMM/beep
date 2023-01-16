import sys, time
import qcfractal.interface as ptl
from pathlib import Path
from beep.converge_sampling import sampling

from optparse import OptionParser
usage ='''python %prog [options]

A command line interface to sample a binding site on a water clusters (stored in a
QCFractal DataSet)  with a small molecule or atom. This CLI is part
of the Binding Energy Evaluation Platform (BEEP).
'''
parser = OptionParser(usage=usage)
parser.add_option("--client-address",
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
parser.add_option("--cluster-collection",
                  dest="cluster_collection",
                  help="The QCFractal name of the collection that containes the cluster to be sampled, if the cluster belongs to BEEP this is not necessary  (Default: None)",
                  default=None
)
parser.add_option("--sampling-molecule-name",
                   dest="sampling_mol_name",
                   help="The name of the molecule that samples the cluster as it appears in the sampling molecule collection",
)
parser.add_option("--sampling-molecule-collection",
                   dest="sampling_molecule_collection",
                   help="The name of the QCArchive collection containing molecules or radicals (dafault: Small_molecules)",
                   default="Small_molecules"
)
parser.add_option("--molecule-size",
                  dest="molecule_size",
                  help="The size of the molecule bound to the cluster that will be sampled"
)
parser.add_option("--sampling-shell",
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
parser.add_option("--maximal-binding-sites",
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
parser.add_option("--grid-size",
                  dest="grid_size",
                  type = "str",
                  default= "sparse",
                  help="The size of the grid: dense, normal, sparse (default: sparse)"
)
parser.add_option("-l",
                  "--level-of-theory",
                  dest="level_of_theory",
                  help=
                  "The level of theory in the format: method_basis (default: blyp_def2-svp)",
                  default="blyp_def2-svp"
                 )
parser.add_option("--refinement-level-of-theory",
                  dest="r_level_of_theory",
                  help=
                  "The level of theory for geometry refinement in the format: method_basis (default: hf3c_minix)",
                  default="hf3c_minix"
                 )
parser.add_option("--rmsd-value",
                  dest="rmsd_val",
                  type = "float",
                  default= 0.10,
                  help="Rmsd geometrical criteria, all structure below this value will not be considered as unique. (default: 0.40 angstrom)",
)
parser.add_option("--rmsd-symmetry",
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
parser.add_option("--sampling-tag",
                  dest="tag",
                  default="sampling",
                  help="The tag to used to specify the qcfractal-manager for the sampling optimization  (default: sampling)",
                  )

parser.add_option("-k",
                  "--keyword-id",
                  dest="keyword_id",
                  help="ID of the QC keywords for the OptimizationDataSet specification (default: None)",
                  default=None)
parser.add_option("--purge",
                  dest="purge",
                  type = "float",
                  help="Eliminate points that are to close to each other. Value should be given in Angstrom (default: None)",
                  default=None)

parser.add_option("--noise",
                  action='store_true',
                  dest="noise",
                  help="Add some randomness to the positions of the points."
)
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
cluster_coll = options.cluster_collection
sampling_mol_collection = options.sampling_molecule_collection
sampling_mol_name = options.sampling_mol_name
sampled_mol_size = options.molecule_size
molecule_size = options.molecule_size
s_shell = options.sampling_shell
zenith_angle = options.zenith_angle
max_struct = options.maximal_binding_sites
max_rounds = options.number_of_rounds
noise = options.noise
purge = options.purge

grid_size = options.grid_size

r_lot = options.r_level_of_theory
single_site = True

client = ptl.FractalClient(address=options.client_address, verify=False, username = username, password=password)

# Getting OptimizationDataset of the cluster with the molecule to be sampled.
if not cluster_coll:
    cluster_coll = "_".join(cluster_name.split('_')[:-1])

try:
    ds_opt = client.get_collection("OptimizationDataset", cluster_coll)
except KeyError:
    print("""Collection with structure to be sampled  {} does not exist. Please check name. Exiting...
    """.format(cluster_col))
    sys.exit(1)

# Retriving the cluster molecule object from the dataset
try:
    cluster = ds_opt.get_record(cluster_name, opt_lot).get_final_molecule()
except KeyError:
    print("{} is not optimized at the requested level of theory or does not exist. \n".format(cluster_name))
    sys.exit(1)

# Getting OptimizationDataset of the sampling molecule.
try:
    ds_opt = client.get_collection("OptimizationDataset", sampling_mol_collection )
except KeyError:
    print("""Collection with sampling molecule  {} does not exist. Please check name. Exiting...
    """.format(sampling_mol_collection))
    sys.exit(1)

# Retriving the sampling molecule object from the OptimizationDataset
try:
    sampling_mol = ds_opt.get_record(sampling_mol_name, opt_lot).get_final_molecule()
except KeyError:
    print("{} is not optimized at the requested level of theory or does not exist. \n".format(cluster_name))
    sys.exit(1)

m = r_lot.split('_')[0]
b = r_lot.split('_')[1]

method = m
basis = b

if program == 'terachem':
    method = m.split('-')[0]
    if 'd3' in method:
        kw = ptl.models.KeywordSet(**{"values": {"dftd": "d3", "convthre" : '3.0e-7', "threall" : '1.0e-13', 'dftgrid' : 2,  "scf" : "diis+a"}})
        kw_id = client.add_keywords([kw])[0]
    else:
        kw = ptl.models.KeywordSet(**{"values": {"convthre" : '3.0e-7', "threall" : '1.0e-13', "scf" : "diis+a"}})
        kw_id = client.add_keywords([kw])[0]



add_ref_spec = {'name': m+'_'+b,
        'description': 'Geometric + '+program+'/'+m+'/'+b,
                'optimization_spec': {'program': 'geometric', 'keywords': {'converge' : ["set" , "gau_tight"], 'maxiter': 150}},
        'qc_spec': {'driver': 'gradient',
        'method': method,
        'basis': basis,
        'keywords': kw_id,
        'program': program}}


print("Processing cluster: {}".format(cluster_name))

opt_dset_name = cluster_name+"+"+sampling_mol_name

try:
    ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
    c = len(ds_opt.status(collapse=False))
except KeyError:
    pass

out_file = Path("./site_finder/"+str(sampling_mol_name)+"/"+ cluster_name + "/out_sampl.dat")

if not out_file.is_file():
    out_file.parent.mkdir(parents=True, exist_ok=True)

with open(out_file, 'w') as f:
    f.write('''        Welcome to the single site sampler !

Description: The molecular sampler converger optimizes configurations
of a small molecule around a bound target  molecule until no more unique stationary
points are found.

Author: svogt
Data: 12/12/2022
version: 0.1.0

'''
    )
s_conv = sampling(method, basis, program, tag, kw_id, opt_dset_name, opt_lot, rmsd_symm,
                  rmsd_val, sampling_mol, cluster, out_file, client,
                  max_struct=max_struct, grid_size=grid_size,  sampled_mol_size = sampled_mol_size,
                  zenith_angle=zenith_angle, max_rounds=max_rounds,
                  single_site=single_site, sampling_shell = s_shell, noise = noise, purge = purge)

if s_conv:
   ds_opt = client.get_collection("OptimizationDataset", opt_dset_name)
   ds_opt.add_specification(**add_ref_spec,overwrite=True)
   ds_opt.save()
   c=ds_opt.compute(m+"_"+b, tag="refinement_tera")
   with open(out_file, 'a') as f:
       f.write('''
       Sending  {} refinement optimizations at {} level. and exiting. Thank you!'''.format(c,r_lot))
else:
   print("An error occured. Check the output in the site-finder folder")
   sys.exit(1)
