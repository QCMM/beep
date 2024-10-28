#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash benchmark_energy_input.sh"
}

# Check if the "--help" flag is provided
if [ "$1" == "--help" ]; then
    # Display help message and exit
    python ../workflows/launch_energy_benchmark.py --help
    exit 0
fi

# Python program invocation
python ../workflows/launch_energy_benchmark.py \
    --client_address localhost:7777 \
    --username '' \
    --password ''  \
    --small-molecule-collection small_molecules \
    --benchmark-structures 'co_w2_0001' 'co_w2_0007' 'co_w3_0001' 'co_w3_0004' 'co_w3_0025' \
    --molecule CO \
    --surface-model-collection small_water \
    --reference-geometry-level-of-theory 'DF-CCSD(T)-F12' 'cc-pVDZ-F12' 'molpro' \
    --optimization-level-of-theory MPWB1K-D3BJ_def2-tzvp  HF3C_MINIX  \

