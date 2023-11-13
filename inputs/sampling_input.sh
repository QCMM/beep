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
    --surface-model-collection water_22 \
    --small-molecule-collection small_molecules \
    --sampling-shell 3.0 \
    --sampling-condition normal \
    --sampling-level-of-theory blyp-d3 def2-svp terachem\
    --refinement-level-of-theory bhandhlyp def2-svp psi4\
    --rmsd-value 0.4 \
    --rmsd-symmetry \
    --store-initial-structures \
    --sampling-tag sampling \
    --total-binding-sites 250 \
    --keyword-id 1
