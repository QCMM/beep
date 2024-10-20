#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash binding_energy.sh"
}

# Check if the "--help" flag is provided
if [ "$1" == "--help" ]; then
    # Display help message and exit
    python ../workflows/launch_be_hess.py --help \
    exit 0
fi

# Usage message
echo "Usage: bash binding_energy.sh path/to/scratch path/to/outputfile"

# Python program invocation
python ../workflows/launch_be_hess.py \
    --client-address localhost:7777 \
    --username 'username' \
    --password "IloveBeep" \
    --surface-model-collection  water_beep_1 \
    --small-molecule-collection hydrocarbons_beep-1 \
    --molecule CH4 \
    --opt-level-of-theory mpwb1k-d3bj_def2-tzvp \
    --level-of-theory WPBE-D3BJ_def2-tzvp PW6B95-D3BJ_def2-tzvp MPWB1K-D3BJ_def2-tzvp \
    --exclude-cluster w8_01 w7_03\
    --program psi4 \
    --energy-tag beep_be_small \
    --hessian-tag be_hessian \
    --hessian-clusters w6_01 w6_02\

