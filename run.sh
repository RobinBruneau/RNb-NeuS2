#!/bin/bash

# Check if folder path is provided as argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [--no-albedo | --l2]"
    exit 1
fi

# Assign folder path to a variable
case="$1"

# Initialize flags variable
flags=""
if [ $# -ge 2 ]; then
    case "$2" in
        --no-albedo)
            flags+=" --no-albedo"
            ;;
        --ltwo)
            flags+=" --ltwo"
            ;;
        *)
            echo "Invalid option: $2"
            echo "Usage: $0 <folder_path> [--no-albedo | --l2]"
            exit 1
            ;;
    esac
fi

iterWarmupMask="5000"
iterWarmupLightGlobal="15000"
iterLightOptimal="25000"
resolutionMarchingCube="1024"

# Display the command for debugging
echo ./build/testbed --scene "${case}/" --maxiter "${iterWarmupMask}" --save-snapshot --mask-weight 1.0 --no-gui $flags

# Execute the commands with the defined variables
./build/testbed --scene "${case}/" --maxiter "${iterWarmupMask}" --save-snapshot --mask-weight 1.0 --no-gui $flags

./build/testbed --scene "${case}/" --maxiter "${iterWarmupLightGlobal}" --save-snapshot --mask-weight 0.3 --no-gui --snapshot "${case}/snapshot_${iterWarmupMask}.msgpack" $flags

./build/testbed --scene "${case}/" --maxiter "${iterLightOptimal}" --save-snapshot --mask-weight 0.3 --no-gui --snapshot "${case}/snapshot_${iterWarmupLightGlobal}.msgpack" --save-mesh --resolution "${resolutionMarchingCube}" --opti-lights $flags