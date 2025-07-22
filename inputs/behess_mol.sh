#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash binding_energy.sh"
}

# Check if the "--help" flag is provided
if [ "$1" == "--help" ]; then
    # Display help message and exit
    python ../workflows/launch_be_hess.py --help
    exit 0
fi

# Usage message
echo "Usage: bash binding_energy.sh path/to/scratch path/to/outputfile"

# Python program invocation
python ../workflows/launch_be_hess.py \
    --client-address localhost:7777 \
    --username  \
    --password  \
    --surface-model-collection SURFACE_MODEL_NAME \
    --small-molecule-collection MOLCULE_COLLECTION_NAME \
    --molecule MOLECULE \
    --opt-level-of-theory mpwb1k-d3bj_def2-tzvpd \
    --program psi4 \
    --energy-tag beep_be_rads \
    --hessian-tag hess_rads \
    --hessian-clusters w5_01 w6_01 w6_02  w6_03 w7_01 \
    --level-of-theory WPBE-D3BJ_def2-tzvpd PW6B95-D3BJ_def2-tzvpd MPWB1K-D3BJ_def2-tzvpd WB97X-V_def2-tzvpd \

