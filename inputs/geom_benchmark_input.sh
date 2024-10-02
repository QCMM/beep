#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash sampling.sh"
}

# Check if the "--help" flag is provided
if [ "$1" == "--help" ]; then
    # Display help message and exit
    python ../scripts/launch_geom_benchmark.py --help
    exit 0
fi

# Python program invocation
python ../scripts/launch_geom_benchmark.py \
    --client_address "152.74.10.245:7777" \
    --username svogt \
    --password 7kyRT-Mrow3jH0Lg6b9YIhEjAcvU9EpFBb9ouMClU5g  \
    --benchmark-structures hf_W3_0001 hf_W3_0004 hf_W3_0039 hf_W2_0001  \
    --small-molecule-collection small_molecules \
    --molecule HF \
    --surface-model-collection small_water \
    --reference-geometry-level-of-theory 'DF-CCSD(T)-F12' 'cc-pVDZ-F12' 'molpro' \
    --dft-optimization-program psi4 \
