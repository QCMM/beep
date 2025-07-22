#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash sampling.sh"
}

# Check if the "--help" flag is provided
if [ "$1" == "--help" ]; then
    # Display help message and exit
python ../workflows/launch_sampling.py --help
    exit 0
fi

# Python program invocation
python ../workflows/launch_sampling.py \
    --client_address localhost:7777 \
    --username '' \
    --password "" \
    --molecule MOLNAME \
    --surface-model-collection SURFACE_MODEL_NAME \
    --small-molecule-collection MOLECULE_COLLECTION_NAME \
    --sampling-shell 2.0 \
    --sampling-condition fine \
    --refinement-level-of-theory mpwb1k-d3bj def2-tzvpd psi4 \
    --rmsd-value 0.4 \
    --rmsd-symmetry \
    --store-initial-structures \
    --sampling-tag sampling \
    --refinement-tag ref_rads \
    --total-binding-sites 75 \
    #--keyword-id 3

