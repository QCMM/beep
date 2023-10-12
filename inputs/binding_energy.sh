#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash binding_energy.sh"
}

# Check if the "--help" flag is provided
if [ "$1" == "--help" ]; then
    # Display help message and exit
    python ../scripts/launch_energy.py --help
    exit 0
fi


# Usage message
echo "Usage: bash binding_energy.sh path/to/scratch path/to/outputfile"

# Python program invocation
python ../scripts/launch_energy.py \
    --client_address 152.74.10.245:7777 \
    --username '' \
    --password "" \
    --surface_model_collection Water_22 \
    --small_molecule_collection Small_molecules \
    --molecule HF \
    --level_of_theory wpbe-d3bj_def2-tzvp \
    --opt_level_of_theory hf3c_minix \
    --keyword_id None \
    --hessian_compute 1 \
    --program psi4 \
    --energy_tag energies \
    --hessian_tag hessian

