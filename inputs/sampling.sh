#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash sampling.sh path/to/scratch path/to/outputfile"
}

# Check if the "--help" flag is provided
if [ "$1" == "--help" ]; then
    # Display help message and exit
    python ../scripts/launch_sampling.py --help
    exit 0
fi

# Python program invocation
python ../scripts/launch_sampling.py \
    --client_address 152.74.10.245:7777 \
    --username '' \
    --password "" \
    --molecule HF \
    --surface_model_collection Water_22 \
    --small_molecule_collection Small_molecules \
    --molecules_per_round 10 \
    --sampling_shell 2.0 \
    --maximal_binding_sites 21 \
    --level_of_theory HF3c_MINIX \
    --refinement_level_of_theory BHANDHLYP_def2-tzvp \
    --rmsd_value 0.4 \
    --rmsd_symmetry \
    --program psi4 \
    --sampling_tag sampling \
