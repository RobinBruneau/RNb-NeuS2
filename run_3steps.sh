#!/bin/bash

# Check if folder path is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

# Assign folder path to a variable
case="$1"

# Execute commands with the provided folder path
./build/testbed --mode nerf --scene "${case}/NeuS2/NeuS2_l60/" \
    --maxiter 5000 --save-mesh --save-snapshot --mask-weight 1.0 --no-gui

./build/testbed --mode nerf --scene "${case}/NeuS2/NeuS2_l60/" \
    --maxiter 15000 --save-mesh --save-snapshot --mask-weight 0.3 \
    --snapshot "${case}/NeuS2/NeuS2_l60/snapshot_5000.msgpack" --no-gui

./build/testbed --mode nerf --opti-lights --scene "${case}/NeuS2/NeuS2_lopti/" \
    --maxiter 25000 --save-mesh --save-snapshot --mask-weight 0.3 \
    --snapshot "${case}/NeuS2/NeuS2_l60/snapshot_15000.msgpack" --no-gui

# Create result directory if not exists
mkdir -p "${case}/result/"

# Copy files to the result directory
cp "${case}/NeuS2/NeuS2_l60/mesh_5000_.obj" "${case}/result/mesh_5000.obj"
cp "${case}/NeuS2/NeuS2_l60/mesh_15000_.obj" "${case}/result/mesh_15000.obj"
cp "${case}/NeuS2/NeuS2_lopti/mesh_25000_.obj" "${case}/result/mesh_25000.obj"
