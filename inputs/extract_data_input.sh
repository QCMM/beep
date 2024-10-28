#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: extract_data_input.sh"
}

# Check if the "--help" flag is provided
if [ "$1" = "--help" ]; then
    # Display help message and exit
    python ../workflows/launch_extract_be_data.py --help
    exit 0
fi

# Python program invocation with updated parameters including hessian clusters
python ..//workflows/launch_extract_be_data.py \
    --server-address localhost:7777 \
    --opt-method mpwb1k-d3bj_def2-tzvp \
    --be-methods PW6B95-D3BJ WPBE-D3BJ MPWB1K-D3BJ \
    --basis def2-tzvp \
    --mol-coll-name hydrocarbons_beep-1 \
    --surface-model water_beep_1 \
    --molecules CH4 \
    --hessian-clusters W6_01 W6_02 W7_01 W7_02 W8_01 \
    --scale-factor 0.958 \
    --be-range -0.1 -28.0 \
    --exclude-clusters w8_01 w7_03 \

