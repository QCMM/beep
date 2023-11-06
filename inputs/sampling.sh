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
    --surface_model_collection Small_water \
    --small_molecule_collection Small_molecules \
    --sampling_shell 2.5 \
    --sampling_condition normal \
    --level_of_theory pbeh3c_def2-msvp \
    --refinement_level_of_theory bhandhlyp_def2-tzvp \
    --rmsd_value 0.4 \
    --program psi4 \
    --sampling_tag test_sampling \
    --rmsd_symmetry \
