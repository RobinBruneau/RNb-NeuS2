#!/bin/bash

# Check if folder path is provided as argument
if [ $# -ne 2 ]; then
    echo "Usage: $0 <folder_path> <--no-albedo or none>"
    exit 1
fi

# Assign folder path to a variable
case="$1"

local flags=""
[ "$2" == "--no-albedo" ] && flags+=" --no-albedo"

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
