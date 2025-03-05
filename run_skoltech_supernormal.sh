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

lockfile_GT="/tmp/run_skoltech3d_GT.lock"
if [ -f "$lockfile_GT" ]; then
    echo "RNb_GT is running"
    exit 1
fi

lockfile_rnb="/tmp/run_skoltech3d.lock"
if [ -f "$lockfile_rnb" ]; then
    echo "RNb is running"
    exit 1
fi

lockfile_sn="/tmp/run_skoltech3d_sn.lock"
if [ -f "$lockfile_sn" ]; then
    echo "SN is running"
    exit 1
fi

touch $lockfile_sn
trap "rm -f $lockfile_sn" EXIT

# Run Original Supernormal
PYTHON_PATH=/mnt/sdc1/bbrument/anaconda3/envs/sn/bin
SUPERNORMAL_PATH=/home/bbrument/dev/SuperNormal

data_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/unimsps-16bits"
# results_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/results/supernormal_unimsps_16bits"
eval_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval"
method="sn/unimsps-16bits"

exps=1
# n_views="100 50 20 10 5 2"
n_views="50"
datasets="dragon golden_snail plush_bear jin_chan green_carved_pot moon_pillow painted_cup red_ceramic_fish painted_samovar green_tea_boxes blue_boxing_gloves golden_bust small_wooden_chessboard amber_vase green_bucket white_human_skull orange_mini_vacuum pink_boot skate white_castle_land wooden_trex"

for exp in $(seq 1 $exps)
do
    for dataset in $datasets
    do

        if [ ! -f "$data_dir/$dataset/normal/0099.png" ] || [ ! -f "$data_dir/$dataset/mask/0099.png" ]; then
            echo "Dataset $dataset does not exist"
            continue
        fi

        for n_view in $n_views
        do

            output_path="$eval_dir/$dataset/$method/nbv-$n_view/nbit-25000/norm2/exp$exp/results_raw"
            if [ -f "$output_path/${dataset}_sn_unimsps-16bits_nbv-${n_view}_nbit-25000_norm2_exp${exp}.ply" ]; then
                echo "Already computed $dataset with $n_view views with L2 loss"
                continue
            fi

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

            if [ ! -f "$output_path" ]; then

                lockfile_run="$output_path/lockfile_run"
                if [ -f "$lockfile_run" ]; then
                    echo "Already running $dataset with $n_view views with L2 loss"
                    continue
                fi
                mkdir -p $output_path
                touch $lockfile_run
                # trap "rm -f $lockfile_run" EXIT

                # Data preparation
                if [ ! -f "$data_dir_view/$dataset/supernormal_data/cameras.npz" ]; then
                    
                    # echo "$PYTHON_PATH/python $SUPERNORMAL_PATH/data_capture_and_preprocessing/convert_idr_normal_map.py --idr_dir $data_dir_view/$dataset --data_dir $data_dir_view/$dataset/supernormal_data --mask_certainty_name mask"
                    $PYTHON_PATH/python $SUPERNORMAL_PATH/data_capture_and_preprocessing/convert_idr_normal_map.py \
                        --idr_dir "$data_dir_view/$dataset" \
                        --data_dir "$data_dir_view/$dataset/supernormal_data" \
                        --mask_certainty_name "mask"
                fi

                # Run original Supernormal
                # echo "$PYTHON_PATH/python $SUPERNORMAL_PATH/exp_runner.py --conf $SUPERNORMAL_PATH/config/skoltech3d.conf --obj_name $data_dir_view/$dataset/supernormal_data"
                $PYTHON_PATH/python $SUPERNORMAL_PATH/exp_runner.py \
                    --conf $SUPERNORMAL_PATH/config/skoltech3d.conf \
                    --obj_name "$data_dir_view/$dataset/supernormal_data" \
                    --resolution 768

                # Move results
                # echo "mv $data_dir_view/$dataset/supernormal_data/exp/exp_*/meshes_validation/iter_00025000.ply $output_path/${dataset}_sn_unimsps-16bits_nbv-${n_view}_nbit-25000_norm2_exp${exp}.ply"
                mv $data_dir_view/$dataset/supernormal_data/exp/exp_*/meshes_validation/iter_00025000.ply $output_path/${dataset}_sn_unimsps-16bits_nbv-${n_view}_nbit-25000_norm2_exp${exp}.ply

                rm -f $lockfile_run
    
            fi

            if [ -d "$data_dir_view/$dataset/supernormal_data" ]; then
                echo "Cleaning up $dataset with $n_view views"
                rm -rf "$data_dir_view/$dataset/supernormal_data"
            fi

        done
    done
done
