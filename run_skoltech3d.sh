#!/bin/bash

lockfile_sdm="/tmp/run_sdm.lock"
if [ -f "$lockfile_sdm" ]; then
    echo "SDM running"
    exit 1
fi

lockfile_neus2="/tmp/run_skoltech3d_neus2.lock"
if [ -f "$lockfile_neus2" ]; then
    echo "NeuS2 is running"
    exit 1
fi

lockfile_rnb="/tmp/run_skoltech3d.lock"
if [ -f "$lockfile_rnb" ]; then
    echo "RNb is running"
    exit 1
fi

touch $lockfile_rnb
trap "rm -f $lockfile_rnb" EXIT

data_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/unimsps"
results_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/results/unimsps"

exps=1
n_views="100 50 20 10 5 2"
datasets="dragon golden_snail plush_bear jin_chan green_carved_pot moon_pillow painted_cup red_ceramic_fish painted_samovar green_tea_boxes blue_boxing_gloves golden_bust small_wooden_chessboard amber_vase green_bucket white_human_skull orange_mini_vacuum pink_boot skate white_castle_land wooden_trex"
# for exp in $(seq 1 $exps)
# do
#     for dataset in $datasets
#     do

#         if [ ! -f "$data_dir/$dataset/normal/0099.png" ] && [ ! -f "$data_dir/$dataset/mask/0099.png" ]; then
#             echo "Dataset $dataset does not exist"
#             continue
#         fi

#         for n_view in $n_views
#         do 

#             if [ $n_view == "100" ]; then
#                 data_dir_view=$data_dir
#             else
#                 data_dir_view="$data_dir-$n_view"

#                 if [ $n_view == "50" ]; then
#                     /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98
#                 elif [ $n_view == "20" ]; then
#                     /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 5 10 16 21 26 32 37 42 47 53 58 63 68 74 79 84 89 95 99
#                 elif [ $n_view == "10" ]; then
#                     /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 11 22 33 44 55 66 77 88 99
#                 elif [ $n_view == "5" ]; then
#                     /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 25 50 75 99
#                 elif [ $n_view == "2" ]; then
#                     /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 99
#                 fi
#             fi

#             if [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/n_views-$n_view/exp$exp/mesh_50000_.obj" ]; then
#                 continue
#             fi

#             echo "Processing $dataset with $n_view views"

#             if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
#                 echo "Preprocessing $dataset with $n_view views"
#                 /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
#             fi

#             output_path="$results_dir/$dataset/supernormal/wosnap2center/norm1/n_views-$n_view/exp$exp"
#             if [ ! -f "$output_path/mesh_50000_.obj" ]; then

#                 echo "Running $dataset with $n_view views with L1 loss"

#                 ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter 50000 --supernormal
                
#                 mkdir -p $output_path
#                 mv $data_dir_view/$dataset/rnbneus_data/command.log $output_path
#                 mv $data_dir_view/$dataset/rnbneus_data/run.sh.log $output_path
#                 mv $data_dir_view/$dataset/rnbneus_data/mesh_*_.obj $output_path
#             fi

#             output_path="$results_dir/$dataset/supernormal/wosnap2center/norm2/n_views-$n_view/exp$exp"
#             if [ ! -f "$output_path/mesh_50000_.obj" ]; then

#                 echo "Running $dataset with $n_view views with L2 loss"

                
#                 ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter 50000 --ltwo --supernormal
#                 mkdir -p $output_path
#                 mv $data_dir_view/$dataset/rnbneus_data/command.log $output_path
#                 mv $data_dir_view/$dataset/rnbneus_data/run.sh.log $output_path
#                 mv $data_dir_view/$dataset/rnbneus_data/mesh_*_.obj $output_path
#             fi

#             if [ -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then

#                 echo "Cleaning up $dataset with $n_view views"

#                 rm -rf "$data_dir_view/$dataset/rnbneus_data"
#             fi
#         done
#     done
# done


data_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/sdmunips"
results_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/results/sdmunips"
n_views="20"
# datasets="golden_snail plush_bear green_carved_pot moon_pillow red_ceramic_fish"
datasets="dragon golden_snail plush_bear jin_chan green_carved_pot moon_pillow painted_cup red_ceramic_fish painted_samovar green_tea_boxes blue_boxing_gloves golden_bust small_wooden_chessboard amber_vase green_bucket white_human_skull orange_mini_vacuum pink_boot skate white_castle_land wooden_trex"

for exp in $(seq 1 $exps)
do
    for dataset in $datasets
    do

        if [ ! -f "$data_dir/$dataset/normal/0099.png" ]; then
            echo "Dataset $dataset does not exist"
            continue
        fi

        for n_view in $n_views
        do 

            if [ $n_view == "100" ]; then
                data_dir_view=$data_dir
            else
                data_dir_view="$data_dir-$n_view"

                if [ $n_view == "50" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98
                elif [ $n_view == "20" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 5 10 16 21 26 32 37 42 47 53 58 63 68 74 79 84 89 95 99
                elif [ $n_view == "10" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 11 22 33 44 55 66 77 88 99
                elif [ $n_view == "5" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 25 50 75 99
                elif [ $n_view == "2" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 99
                fi
            fi

            if [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/worgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/worgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/wrgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/wrgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ]; then
                continue
            fi

            if [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm1/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm1_woalbedo_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm2/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm2_woalbedo_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm1/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm1_walbedo_worgbplus_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm2/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm2_walbedo_worgbplus_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm1/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm1_walbedo_wrgbplus_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm2/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm2_walbedo_wrgbplus_exp${exp}_cleaned.ply" ]; then
                echo "Dataset $dataset with $n_view views already processed"
                continue
            fi

            echo "Processing $dataset with $n_view views"

            output_path="$results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp"
            if [ ! -f "$output_path/mesh_50000_.obj" ]; then

                if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                    echo "Preprocessing $dataset with $n_view views"
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                fi

                echo "Running $dataset with $n_view views with L1 loss and no albedo"
                ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter 50000 --supernormal
                
                mkdir -p $output_path
                mv $data_dir_view/$dataset/rnbneus_data/command.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data/run.sh.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data/mesh_*_.obj $output_path
            fi

            output_path="$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/worgbplus/n_views-$n_view/exp$exp"
            if [ ! -f "$output_path/mesh_50000_.obj" ]; then

                if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                    echo "Preprocessing $dataset with $n_view views"
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                fi

                echo "Running $dataset with $n_view views with L1 loss and albedo"
                if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path $results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj
                fi

                ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter 50000 --supernormal

                mkdir -p $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_*_.obj $output_path
            fi

            output_path="$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/wrgbplus/n_views-$n_view/exp$exp"
            if [ ! -f "$output_path/mesh_50000_.obj" ]; then

                if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                    echo "Preprocessing $dataset with $n_view views"
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                fi

                echo "Running $dataset with $n_view views with L1 loss and albedo and rgb+"
                if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path $results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj
                fi

                ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter 50000 --supernormal --rgbplus

                mkdir -p $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_*_.obj $output_path
            fi

            output_path="$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp"
            if [ ! -f "$output_path/mesh_50000_.obj" ]; then

                if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                    echo "Preprocessing $dataset with $n_view views"
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                fi

                echo "Running $dataset with $n_view views with L2 loss and no albedo"
                ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter 50000 --ltwo --supernormal
                
                mkdir -p $output_path
                mv $data_dir_view/$dataset/rnbneus_data/command.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data/run.sh.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data/mesh_*_.obj $output_path
            fi

            output_path="$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/worgbplus/n_views-$n_view/exp$exp"
            if [ ! -f "$output_path/mesh_50000_.obj" ]; then

                if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                    echo "Preprocessing $dataset with $n_view views"
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                fi

                echo "Running $dataset with $n_view views with L2 loss and albedo"
                if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path $results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj
                fi

                ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter 50000 --ltwo --supernormal

                mkdir -p $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_*_.obj $output_path
            fi

            output_path="$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/wrgbplus/n_views-$n_view/exp$exp"
            if [ ! -f "$output_path/mesh_50000_.obj" ]; then

                if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                    echo "Preprocessing $dataset with $n_view views"
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                fi

                echo "Running $dataset with $n_view views with L2 loss and albedo"
                if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path $results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj
                fi

                ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter 50000 --ltwo --supernormal --rgbplus

                mkdir -p $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_*_.obj $output_path
            fi

            if [ -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                echo "Cleaning up $dataset with $n_view views"
                rm -rf "$data_dir_view/$dataset/rnbneus_data"
            fi

            if [ -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                echo "Cleaning up $dataset with $n_view views"
                rm -rf "$data_dir_view/$dataset/rnbneus_data-albedoscaled"
            fi
        done
    done
done
