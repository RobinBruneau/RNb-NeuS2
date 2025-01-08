#!/bin/bash

# Check if folder path is provided as argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [--num-iter <num_iter> | --res <resolution> | --disable-snap-to-center | --no-albedo]"
    exit 1
fi

# Assign folder path to a variable
case="$1"

# Log the run.sh command and the run.sh script in the output folder
cp run_supernormal.sh "$case/run.sh.log"
echo "./run_supernormal.sh $@" > "$case/command.log"

# Handle optional arguments
flags=""
while [ $# -gt 1 ]; do
    case "$2" in
        --num-iter)
            num_iter="$3"
            flags="$flags --num-iter $num_iter"
            shift
            ;;
        --res)
            resolution="$3"
            shift
            ;;
        --disable-snap-to-center)
            flags="$flags --disable-snap-to-center"
            ;;
        --no-albedo)
            flags="$flags --no-albedo"
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
./build/testbed --scene "${case}/" --no-gui --maxiter "${num_iter}" --save-snapshot --save-mesh --mask-weight 1.0 --resolution ${resolutionMarchingCube} --supernormal $flags