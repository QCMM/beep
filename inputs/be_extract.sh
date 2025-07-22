#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash sampling.sh"
}

# Check if the "--help" flag is provided
if [ "$1" = "--help" ]; then
    # Display help message and exit
    python ../workflows/launch_extract_be_data.py --help
    exit 0
fi

# Python program invocation with updated parameters including hessian clusters
python ../workflows/launch_extract_be_data.py \
    --server-address localhost:7777 \
    --opt-method mpwb1k-d3bj_def2-tzvpd \
    --be-methods WPBE-D3BJ PW6B95-D3BJ MPWB1K-D3BJ WB97X-V  \
    --basis def2-tzvpd \
    --mol-coll-name MOLECULE_COLLECTION_NAME \
    --surface-model wx-beep1 \
    --molecules MOL1 MOL2 MOL3...  \
    --hessian-clusters w6_01 w6_02 w6_03 w7_01 w5_01  \
    --be-range -0.1 -28.0 \
    --scale-factor 0.954 \
    #--no-zpve
