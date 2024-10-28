#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash sampling_input.sh"
}

# Check if the "--help" flag is provided
if [ "$1" == "--help" ]; then
    # Display help message and exit
    python ../workflows/launch_sampling.py --help \
    exit 0
fi

# Python program invocation
python ../workflows/launch_sampling.py \
    --client_address localhost:7777 \
    --username '' \
    --password "" \
    --molecule CH4 \
    --surface-model-collection water_beep_1 \
    --small-molecule-collection hydrocarbons_beep-1 \
    --sampling-shell 3.0 \
    --sampling-condition normal \
    --sampling-level-of-theory blyp-d3 def2-svp terachem\
    --refinement-level-of-theory pbe0-d3bj def2-tzpv psi4\
    --refinement-tag refinement \
    --rmsd-value 0.4 \
    --rmsd-symmetry \
    --store-initial-structures \
    --sampling-tag sampling \
    --total-binding-sites 50 \
    --keyword-id 1 

