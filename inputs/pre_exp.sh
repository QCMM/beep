#!/bin/bash

# Function to display usage message
show_usage() {
    echo "Usage: bash pre_exp.sh"
}

# Check if the "--help" flag is provided
if [ "$1" = "--help" ]; then
    # Display help message and exit
    python ../workflows/launch_pre_exp.py --help
    exit 0
fi

python ../workflows/launch_pre_exp.py \
    --client-address localhost:7777 \
    --username '' \
    --password '' \
    --molecule-collection '' \
    --level-of-theory '' \
    --range-of-temperature \
