#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash sampling.sh"
}

# Check if the "--help" flag is provided
if [ "$1" == "--help" ]; then
    # Display help message and exit
    python ../benchmark/launch_benchmark.py --help
    exit 0
fi

# Python program invocation
python ../benchmark/launch_benchmark.py \
    --client_address "152.74.10.245:7777" \
    --username svogt \
    --password 7kyRT-Mrow3jH0Lg6b9YIhEjAcvU9EpFBb9ouMClU5g  \
    --benchmark-structures nh3_w2_0001 nh3_w3_0002 nh3_w3_0007 \
    --basis-set def2-svp \
    --small-molecule-collection small_molecules \
    --molecule NH3 \
    --surface-model-collection small_water \
    --reference-geometry-level-of-theory 'DF-CCSD(T)-F12' 'cc-pVDZ-F12' 'molpro' \
    --reference-energy-level-of-theory 'ccsd(t)' 'cbs' 'psi4' \
    --rmsd-value 0.15 \
    --rmsd-symmetry

