#!/bin/bash

# Check if folder path is provided as argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [--supernormal | --num-iter <num_iter> | --res <resolution>]"
    exit 1
fi

# Assign folder path to a variable
case="$1"

# Handle optional arguments
flags=""
while [ $# -gt 1 ]; do
    case "$2" in
        --supernormal)
            flags="$flags --supernormal"
            ;;
        --num-iter)
            num_iter="$3"
            flags="$flags --num-iter $num_iter"
            shift
            ;;
        --res)
            resolution="$3"
            shift
            ;;
        *)
            echo "Unknown option: $2"
            exit 1
            ;;
    esac
    shift
done

# If --num-iter is not provided, set it to 10000
if [ -z "$num_iter" ]; then
    num_iter=10000
fi

# Remove num-iter flag from flags
flags=$(echo "$flags" | sed 's/--num-iter [0-9]*//')

resolutionMarchingCube=${resolution:-1024}

# Execute the commands with the defined variables
./build/testbed --scene "${case}/" --no-gui --maxiter "${num_iter}" --save-snapshot --save-mesh --mask-weight 1.0 --resolution ${resolutionMarchingCube} --disable-snap-to-center --supernormal $flags