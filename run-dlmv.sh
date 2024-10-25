#!/bin/bash


scenes="bearPNG buddhaPNG cowPNG pot2PNG readingPNG"
# for scene in $scenes; do
#     output_path=data/DLMV_ALBEDO_SDM/$scene
#     python scripts/preprocess.py --folder ${output_path} \
#         --mask_certainty_name "mask_certainty" \
#         --exp_name "RNb-NeuS2-withmaskcertainty"

#     python scripts/preprocess.py --folder ${output_path} \
#         --mask_certainty_name "mask" \
#         --exp_name "RNb-NeuS2-withoutmaskcertainty"
# done

for scene in $scenes; do
    output_path=data/DLMV_ALBEDO_SDM/$scene
    if [ ! -f "${output_path}/RNb-NeuS2-withmaskcertainty-albedoscaled/mesh_25000_.obj" ]; then
        ./run.sh ${output_path}/RNb-NeuS2-withmaskcertainty --res 1024 --num-iter 25000 --scale-albedo --ltwo
    fi
done

for scene in $scenes; do
    output_path=data/DLMV_ALBEDO_SDM/$scene
    if [ ! -f "${output_path}/RNb-NeuS2-withoutmaskcertainty-albedoscaled/mesh_25000_.obj" ]; then
        ./run.sh ${output_path}/RNb-NeuS2-withoutmaskcertainty --res 1024 --num-iter 25000 --scale-albedo --ltwo
    fi
done
