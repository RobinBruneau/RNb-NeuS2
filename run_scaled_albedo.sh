#!/bin/bash

# Assign folder path to a variable
case="$1"

# Launch without albedo
./run.sh "$case/" --no-albedo

# Launch with scaled albedos
python scripts/scale_albedos.py --folder "$case/"
path=$(dirname "$case")
folder=$(basename "$case")
folder="${folder}-albedoscaled"
./run.sh "$path/$folder/"