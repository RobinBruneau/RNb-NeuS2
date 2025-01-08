#!/bin/bash

# Check if folder path is provided as argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [--no-albedo | --ltwo | --scale-albedo | --num-iter <num_iter> | --res <resolution> | --disable-snap-to-center | --not-opti-lights | --rgbplus | --snapshot <snapshot_path> | --iter-opti-lights <iter_opti_lights> | --supernormal | --save-each <save_each> ]"
    exit 1
fi

# Assign folder path to a variable
case="$1"

# Log the run.sh command and the run.sh script in the output folder
cp run.sh "$case/run.sh.log"
echo "./run.sh $@" > "$case/command.log"

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
            flags="$flags --res $resolution"
            shift
            ;;
        --disable-snap-to-center)
            flags="$flags --disable-snap-to-center"
            ;;
        --not-opti-lights)
            flags="$flags --not-opti-lights"
            not_opti_lights=true
            ;;
        --rgbplus)
            flags="$flags --rgbplus"
            ;;
        --snapshot)
            flags="$flags --snapshot $3"
            shift
            ;;
        --iter-opti-lights)
            iter_opti_lights="$3"
            flags="$flags --iter-opti-lights $iter_opti_lights"
            shift
            ;;
        --supernormal)
            flags="$flags --supernormal"
            supernormal=true
            ;;
        --save-each)
            flags="$flags --save-each $3"
            shift
            ;;
        *)
            echo "Unknown option: $2"
            exit 1
            ;;
    esac
    shift
done

# Default values for optional arguments
num_iter=${num_iter:-3000}
resolution=${resolution:-1024}
iter_opti_lights=${iter_opti_lights:-$(($num_iter/3*2))}
supernormal=${supernormal:-false}
    
# If --scale-albedo is provided, run the albedo scaling steps
if [ "$scale_albedo" = true ]; then
    
    # if scale_albedo is true, then no-albedo flag should not be present
    flags=$(echo "$flags" | sed 's/--no-albedo//')

    # Launch without albedo
    ./run.sh "$case" --no-albedo $flags

    # Launch with scaled albedos
    python scripts/scale_albedos.py --folder "$case"
    path=$(dirname "$case")
    folder=$(basename "$case")
    folder="${folder}-albedoscaled"
    ./run.sh "$path/$folder/" $flags 
    exit 0
fi

# Remove some flags from the flags variable
flags=$(echo "$flags" | sed 's/--num-iter [0-9]*//')
flags=$(echo "$flags" | sed 's/--not-opti-lights//')
flags=$(echo "$flags" | sed 's/--res [0-9]*//')
flags=$(echo "$flags" | sed 's/--iter-opti-lights [0-9]*//')
flags=$(echo "$flags" | sed 's/--supernormal//')

# If --supernormal is provided, run the supernormal 
if [ "$supernormal" = false ]; then

    # Execute the commands with the defined variables
    echo "./build/testbed --scene ${case}/ --maxiter ${iter_opti_lights} --save-snapshot --mask-weight 1.0 --no-gui $flags"
    ./build/testbed --scene "${case}/" --maxiter "${iter_opti_lights}" --save-snapshot --mask-weight 1.0 --no-gui $flags

    # Remove the --snapshot flag from flags
    echo "$flags"
    flags=$(echo "$flags" | sed 's/--snapshot [^ ]*//')
    echo "$flags"

    if [ "$not_opti_lights" = true ]; then
        echo "./build/testbed --scene ${case}/ --maxiter ${num_iter} --save-snapshot --mask-weight 1.0 --no-gui --snapshot ${case}/snapshot_${iter_opti_lights}.msgpack --save-mesh --resolution ${resolution} $flags"
        ./build/testbed --scene "${case}/" --maxiter "${num_iter}" --save-snapshot --mask-weight 1.0 --no-gui --snapshot "${case}/snapshot_${iter_opti_lights}.msgpack" --save-mesh --resolution "${resolution}" $flags
        exit 0
    fi

    echo "./build/testbed --scene ${case}/ --maxiter ${num_iter} --save-snapshot --mask-weight 1.0 --no-gui --snapshot ${case}/snapshot_${iter_opti_lights}.msgpack --save-mesh --resolution ${resolution} --opti-lights $flags"
    ./build/testbed --scene "${case}/" --maxiter "${num_iter}" --save-snapshot --mask-weight 1.0 --no-gui --snapshot "${case}/snapshot_${iter_opti_lights}.msgpack" --save-mesh --resolution "${resolution}" --opti-lights $flags
    exit 0

elif [ "$supernormal" = true ]; then

    # Execute the commands with the defined variables
    echo "./build/testbed --scene ${case}/ --maxiter ${num_iter} --save-snapshot --save-mesh --mask-weight 1.0 --no-gui --resolution ${resolution} --supernormal $flags"
    ./build/testbed --scene "${case}/" --maxiter "${num_iter}" --save-snapshot --save-mesh --mask-weight 1.0 --no-gui --resolution ${resolution} --supernormal $flags
    exit 0

fi