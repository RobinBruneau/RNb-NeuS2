#!/bin/bash

# Check if folder path is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

# Assign folder path to a variable
case="$1"

# Execute commands with the provided folder path
./build/testbed --scene "${case}/" \
    --maxiter 5000 --save-mesh --save-snapshot --mask-weight 1.0 --no-albedo --no-gui
mv "${case}/snapshot_5000.msgpack" "${case}/snapshot_5000.msgpack"

./build/testbed --scene "${case}/" \
    --maxiter 15000 --save-mesh --save-snapshot --mask-weight 0.3 \
    --snapshot "${case}/snapshot_5000.msgpack" --no-gui --no-albedo
mv "${case}/snapshot_15000.msgpack" "${case}/snapshot_15000.msgpack"

./build/testbed --opti-lights --scene "${case}/" \
    --maxiter 25000 --save-mesh --save-snapshot --mask-weight 0.3 \
    --snapshot "${case}/snapshot_15000.msgpack" --no-gui --no-albedo
mv "${case}/snapshot_25000.msgpack" "${case}/snapshot_25000.msgpack"
