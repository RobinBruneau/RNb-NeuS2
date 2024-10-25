#!/bin/bash

# Check if folder path is provided as argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [--no-albedo | --ltwo | --scale-albedo | --num-iter <num_iter> | --res <resolution> | --disable-snap-to-center]"
    exit 1
fi

# Assign folder path to a variable
case="$1"

# Log the run.sh command and the run.sh script in the output folder
cp run.sh "$case/run.sh.log"
echo "./run.sh $case $@" > "$case/command.log"

# Handle optional arguments
flags=""
scale_albedo=false
while [ $# -gt 1 ]; do
    case "$2" in
        --no-albedo)
            flags="$flags --no-albedo"
            ;;
        --ltwo)
            flags="$flags --ltwo"
            ;;
        --scale-albedo)
            scale_albedo=true
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
        --disable-snap-to-center)
            flags="$flags --disable-snap-to-center"
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

# If --scale-albedo is provided, run the albedo scaling steps
if [ "$scale_albedo" = true ]; then
    
    # if scale_albedo is true, then no-albedo flag should not be present
    flags=$(echo "$flags" | sed 's/--no-albedo//')

    # Launch without albedo
    ./run.sh "$case" --no-albedo $flags --res 512

    # Launch with scaled albedos
    python scripts/scale_albedos.py --folder "$case"
    path=$(dirname "$case")
    folder=$(basename "$case")
    folder="${folder}-albedoscaled"
    ./run.sh "$path/$folder/" $flags 
    exit 0
fi

# Remove num-iter flag from flags
flags=$(echo "$flags" | sed 's/--num-iter [0-9]*//')

iterLightOptimal=${num_iter}
iterWarmupLightGlobal=$(($iterLightOptimal/5*3))
iterWarmupMask=$(($iterLightOptimal/5))

resolutionMarchingCube=${resolution:-1024}

# Execute the commands with the defined variables
./build/testbed --scene "${case}/" --maxiter "${iterWarmupMask}" --save-snapshot --mask-weight 1.0 --no-gui $flags
./build/testbed --scene "${case}/" --maxiter "${iterWarmupLightGlobal}" --save-snapshot --mask-weight 0.3 --no-gui --snapshot "${case}/snapshot_${iterWarmupMask}.msgpack" $flags
./build/testbed --scene "${case}/" --maxiter "${iterLightOptimal}" --save-snapshot --mask-weight 0.3 --no-gui --snapshot "${case}/snapshot_${iterWarmupLightGlobal}.msgpack" --save-mesh --resolution "${resolutionMarchingCube}" --opti-lights $flags
