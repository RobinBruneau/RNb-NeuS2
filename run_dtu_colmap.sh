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

lockfile_sn="/tmp/run_skoltech3d_sn.lock"
if [ -f "$lockfile_sn" ]; then
    echo "SN is running"
    exit 1
fi

lockfile_rnb="/tmp/run_skoltech3d.lock"
if [ -f "$lockfile_rnb" ]; then
    echo "RNb is running"
    exit 1
fi

lockfile_colmap="/tmp/run_skoltech3d_colmap.lock"
if [ -f "$lockfile_colmap" ]; then
    echo "Colmap is running"
    exit 1
fi

# lockfile_colmap_spsr="/tmp/lockfile_colmap_spsr.lock"
# if [ -f $lockfile_colmap_spsr ]; then
#     echo "Colmap SPSR is running"
#     exit 1
# fi

touch $lockfile_colmap
trap "rm -f $lockfile_colmap" EXIT

PYTHON_PATH="/mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin"

data_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/dtu/data"
eval_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/DTU_EVAL"

exps=1
n_views="64 49 30 20 10 5 4 3 2"
# datasets="scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122"
datasets="scan122 scan118 scan114 scan110 scan106 scan105 scan97 scan83 scan69 scan65 scan63 scan55 scan40 scan37 scan24"
option_loss="norm2 norm1"
method="colmap"
n_iters="50000"

for exp in $(seq 1 $exps)
do
    for dataset in $datasets
    do

        for n_view in $n_views
        do 

            # Count number of images in "$data_dir/$dataset/$method/nbv-${n_view}"
            data_dir_view="$data_dir/$dataset/$method/nbv-${n_view}"
            # echo "$data_dir_view"
            if [ -d "$data_dir_view/normal" ]; then
                num_images=$(ls -1q $data_dir_view/normal/*.png | wc -l)

                if [ $num_images -lt $n_view ]; then # Skip if the number of images is less than the number of views
                    echo "Skipping $dataset with $n_view views because the number of images is less than the number of views"
                    continue
                fi
            else
                echo "Skipping $dataset with $n_view views because the folder does not exist"
                continue
            fi

            if [ -f "$eval_dir/$dataset/rnbv2-sn/colmap/nbv-$n_view/nbit-${n_iters}/norm1/exp$exp/results_raw/${dataset}_rnbv2-sn_colmap_nbv-${n_view}_nbit-${n_iters}_norm1_exp${exp}.obj" ] && [ -f "$eval_dir/$dataset/rnbv2-sn/colmap/nbv-$n_view/nbit-${n_iters}/norm2/exp$exp/results_raw/${dataset}_rnbv2-sn_colmap_nbv-${n_view}_nbit-${n_iters}_norm2_exp${exp}.obj" ]; then
                echo "Already processed $dataset with $n_view views"
                continue
            fi

            if [ -f "$eval_dir/$dataset/rnbv2-sn/colmap/nbv-$n_view/nbit-${n_iters}/norm1/exp$exp/results_cleaned/${dataset}_rnbv2-sn_colmap_nbv-${n_view}_nbit-${n_iters}_norm1_exp${exp}_cleaned.ply" ] && [ -f "$eval_dir/$dataset/rnbv2-sn/colmap/nbv-$n_view/nbit-${n_iters}/norm2/exp$exp/results_cleaned/${dataset}_rnbv2-sn_colmap_nbv-${n_view}_nbit-${n_iters}_norm2_exp${exp}_cleaned.ply" ]; then
                echo "Already cleaned $dataset with $n_view views"
                continue
            fi

            for option in $option_loss
            do

                output_dir="$eval_dir/$dataset/rnbv2-sn/colmap/nbv-$n_view/nbit-${n_iters}/${option}/exp$exp"
                output_path2="$output_dir/results_raw/${dataset}_rnbv2-sn_colmap_nbv-${n_view}_nbit-${n_iters}_${option}_exp${exp}.obj"
                output_path3="$output_dir/results_cleaned/${dataset}_rnbv2-sn_colmap_nbv-${n_view}_nbit-${n_iters}_${option}_exp${exp}_cleaned.ply"
                if [ ! -f "$output_path2" ] && [ ! -f "$output_path3" ]; then

                    lockfile_run="$output_dir/lockfile_run"
                    if [ -f "$lockfile_run" ]; then
                        echo "Already running $dataset with $n_view views with ${option} loss"
                        continue
                    fi
                    mkdir -p $output_dir
                    touch $lockfile_run

                    if [ ! -f "$data_dir_view/rnbneus_data/transform.json" ]; then
                        echo "Preprocessing $dataset with $n_view views"
                        $PYTHON_PATH/python scripts/preprocess.py --folder $data_dir_view --exp_name "rnbneus_data" --mask_certainty_name "mask_certainty"
                    fi

                    echo "Running $dataset with $n_view views with ${option} loss"
                    if [ $option == "norm1" ]; then
                        ./run.sh $data_dir_view/rnbneus_data --no-albedo --res 1024 --disable-snap-to-center --num-iter ${n_iters} --supernormal
                    elif [ $option == "norm2" ]; then
                        ./run.sh $data_dir_view/rnbneus_data --no-albedo --res 1024 --disable-snap-to-center --num-iter ${n_iters} --ltwo --supernormal
                    fi
                    
                    mkdir -p $output_dir/results_raw
                    mv $data_dir_view/rnbneus_data/command.log $output_dir/results_raw
                    mv $data_dir_view/rnbneus_data/run.sh.log $output_dir/results_raw
                    mv $data_dir_view/rnbneus_data/mesh_* $output_path2

                    rm -f $lockfile_run
                fi

            done

            if [ -f "$data_dir_view/rnbneus_data/transform.json" ]; then
                echo "Cleaning up $dataset with $n_view views"
                rm -rf "$data_dir_view/rnbneus_data"
            fi
        done
    done
done