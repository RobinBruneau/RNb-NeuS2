#!/bin/bash

# Check if folder path is provided as argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [--no-albedo | --ltwo | --scale-albedo]"
    exit 1
fi

# Assign folder path to a variable
case="$1"

# Initialize flags variable
flags=""
scale_albedo=false
if [ $# -ge 2 ]; then
    case "$2" in
        --no-albedo)
            flags+=" --no-albedo"
            ;;
        --ltwo)
            flags+=" --ltwo"
            ;;
        --scale-albedo)
            scale_albedo=true
            ;;
        *)
            echo "Invalid option: $2"
            echo "Usage: $0 <folder_path> [--no-albedo | --ltwo | --scale-albedo]"
            exit 1
            ;;
    esac
fi

# If --scale-albedo is provided, run the albedo scaling steps
if [ "$scale_albedo" = true ]; then
    # Launch without albedo
    ./run.sh "$case" --no-albedo

    # Launch with scaled albedos
    python scripts/scale_albedos.py --folder "$case/"
    path=$(dirname "$case")
    folder=$(basename "$case")
    folder="${folder}-albedoscaled"
    ./run.sh "$path/$folder/"
    exit 0
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
