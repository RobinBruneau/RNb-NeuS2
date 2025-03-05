#!/bin/bash

lockfile_rnb="/tmp/run_skoltech3d.lock"
if [ -f "$lockfile_rnb" ]; then
    echo "RNb is running"
    exit 1
fi

lockfile_GT="/tmp/run_skoltech3d_GT.lock"
if [ -f "$lockfile_GT" ]; then
    echo "RNb_GT is running"
    exit 1
fi

touch $lockfile_GT
trap "rm -f $lockfile_GT" EXIT

data_uni_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/unimsps"
data_sdm_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/sdmunips"
data_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/normals_gt"
results_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/results/normals_gt"
eval_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval"

exps=1
n_views="100"
datasets="dragon golden_snail plush_bear jin_chan green_carved_pot moon_pillow painted_cup red_ceramic_fish painted_samovar green_tea_boxes blue_boxing_gloves golden_bust small_wooden_chessboard amber_vase green_bucket white_human_skull orange_mini_vacuum pink_boot skate white_castle_land wooden_trex"

for exp in $(seq 1 $exps)
do
    for dataset in $datasets
    do

        if [ ! -f "$data_dir/$dataset/normal/0099.png" ]; then
            mkdir -p "$data_dir/$dataset"
            mv "$data_uni_dir/$dataset/normal_gt" "$data_dir/$dataset/normal"
            cp -r "$data_sdm_dir/$dataset/mask" "$data_dir/$dataset"
            cp -r "$data_sdm_dir/$dataset/albedo" "$data_dir/$dataset"
            cp "$data_sdm_dir/$dataset/cameras.npz" "$data_dir/$dataset"
        fi

        if [ ! -f "$data_dir/$dataset/normal/0099.png" ]; then
            echo "Dataset $dataset does not exist"
            continue
        fi

        for n_view in $n_views
        do 

            if [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/worgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/wrgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ]; then
                echo "Dataset $dataset with $n_view views already processed"
                continue
            fi

            # if [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/worgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/worgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/wrgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/wrgbplus/n_views-$n_view/exp$exp/mesh_50000_.obj" ]; then
            #     continue
            # fi

            # if [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-50000/norm1/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm1_woalbedo_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm2/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm2_woalbedo_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm1/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm1_walbedo_worgbplus_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm2/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm2_walbedo_worgbplus_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm1/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm1_walbedo_wrgbplus_exp${exp}_cleaned.ply" ] && [ -f "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-50000/norm2/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-50000_norm2_walbedo_wrgbplus_exp${exp}_cleaned.ply" ]; then
            #     echo "Dataset $dataset with $n_view views already processed"
            #     continue
            # fi

            if [ $n_view == "100" ]; then
                data_dir_view=$data_dir
            else
                data_dir_view="$data_dir-$n_view"

                if [ $n_view == "50" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98
                elif [ $n_view == "20" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 5 10 16 21 26 32 37 42 47 53 58 63 68 74 79 84 89 95 99
                elif [ $n_view == "10" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 11 22 32 44 55 66 77 88 99
                elif [ $n_view == "5" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 25 50 75 99
                elif [ $n_view == "2" ]; then
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 99
                fi
            fi

            # echo "Processing $dataset with $n_view views"

            # output_path1="$results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp"
            # if [ ! -f "$output_path1/mesh_50000_.obj" ]; then

            # lockfile_run="$output_path1/lockfile_run"
            # if [ -f "$lockfile_run" ]; then
            #     echo "Already running $dataset with $n_view views with L1 loss and no albedo"
            #     continue
            # fi
            #     mkdir -p $output_path1
            # touch $lockfile_run

            #     if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
            #         echo "Preprocessing $dataset with $n_view views"
            #         /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
            #     fi

            #     echo "Running $dataset with $n_view views with L1 loss and no albedo"
            #     ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter 50000 --supernormal
                
            #     mv $data_dir_view/$dataset/rnbneus_data/command.log $output_path1
            #     mv $data_dir_view/$dataset/rnbneus_data/run.sh.log $output_path1
            #     mv $data_dir_view/$dataset/rnbneus_data/mesh_* $output_path1/mesh_50000_.obj

            #    rm -f $lockfile_run

            # fi

            # output_path1="$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/worgbplus/n_views-$n_view/exp$exp"
            # if [ ! -f "$output_path1/mesh_50000_.obj" ]; then

            # lockfile_run="$output_path1/lockfile_run"
            # if [ -f "$lockfile_run" ]; then
            #     echo "Already running $dataset with $n_view views with L2 loss and albedo and without rgb+"
            #     continue
            # fi
            #     mkdir -p $output_path1
            # touch $lockfile_run

            #     if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
            #         echo "Preprocessing $dataset with $n_view views"
            #         /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
            #     fi

            #     echo "Running $dataset with $n_view views with L1 loss and albedo"
            #     if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
            #         /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path $results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj
            #     fi

            #     ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter 50000 --supernormal

            #     mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path1
            #     mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path1
            #     mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_* $output_path1/mesh_50000_.obj

            #    rm -f $lockfile_run
            # fi

            # output_path1="$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/wrgbplus/n_views-$n_view/exp$exp"
            # if [ ! -f "$output_path1/mesh_50000_.obj" ]; then

            # lockfile_run="$output_path1/lockfile_run"
            # if [ -f "$lockfile_run" ]; then
            #     echo "Already running $dataset with $n_view views with L1 loss and albedo and rgb+"
            #     continue
            # fi
            #     mkdir -p $output_path1
            # touch $lockfile_run

            #     if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
            #         echo "Preprocessing $dataset with $n_view views"
            #         /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
            #     fi

            #     echo "Running $dataset with $n_view views with L1 loss and albedo and rgb+"
            #     if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
            #         /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path $results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj
            #     fi

            #     ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter 50000 --supernormal --rgbplus

            #     mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path1
            #     mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path1
            #     mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_* $output_path1/mesh_50000_.obj

            #    rm -f $lockfile_run
            # fi

            output_path1="$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp"
            output_path2="$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-50000/norm2/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-50000_norm2_woalbedo_exp${exp}.obj"
            if [ ! -f "$output_path1/mesh_50000_.obj" ] && [ ! -f "$output_path2" ]; then

                lockfile_run="$output_path1/lockfile_run"
                if [ -f "$lockfile_run" ]; then
                    echo "Already running $dataset with $n_view views with L2 loss and no albedo"
                    continue
                fi
                mkdir -p $output_path1
                touch $lockfile_run

                if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                    echo "Preprocessing $dataset with $n_view views"
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                fi

                echo "Running $dataset with $n_view views with L2 loss and no albedo" 
                ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter 50000 --ltwo --supernormal
                
                mv $data_dir_view/$dataset/rnbneus_data/command.log $output_path1
                mv $data_dir_view/$dataset/rnbneus_data/run.sh.log $output_path1
                mv $data_dir_view/$dataset/rnbneus_data/mesh_* $output_path1/mesh_50000_.obj

                rm -f $lockfile_run
            fi

            output_path1="$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/worgbplus/n_views-$n_view/exp$exp"
            output_path2="$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-50000/norm2/walbedo/worgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-50000_norm2_walbedo_worgbplus_exp${exp}.obj"
            echo $output_path1
            echo $output_path2
            if [ ! -f "$output_path1/mesh_50000_.obj" ] && [ ! -f "$output_path2" ]; then

                lockfile_run="$output_path1/lockfile_run"
                if [ -f "$lockfile_run" ]; then
                    echo "Already running $dataset with $n_view views with L2 loss and albedo"
                    continue
                fi
                mkdir -p $output_path1
                touch $lockfile_run

                if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                    echo "Preprocessing $dataset with $n_view views"
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                fi

                echo "Running $dataset with $n_view views with L2 loss and albedo"
                if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                    if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-50000/norm2/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-50000_norm2_woalbedo_exp${exp}.obj" ]; then
                        /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-50000/norm2/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-50000_norm2_woalbedo_exp${exp}.obj"
                    elif [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj" ]; then
                        /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj"
                    fi
                fi

                ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter 50000 --ltwo --supernormal

                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path1
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path1
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_* $output_path1/mesh_50000_.obj

                rm -f $lockfile_run
            fi

            output_path1="$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/wrgbplus/n_views-$n_view/exp$exp"
            output_path2="$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-50000/norm2/walbedo/wrgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-50000_norm2_walbedo_wrgbplus_exp${exp}.obj"
            echo $output_path1
            echo $output_path2
            if [ ! -f "$output_path1/mesh_50000_.obj" ] && [ ! -f "$output_path2" ]; then

                lockfile_run="$output_path1/lockfile_run"
                if [ -f "$lockfile_run" ]; then
                    echo "Already running $dataset with $n_view views with L2 loss and albedo and rgb+"
                    continue
                fi
                mkdir -p $output_path1
                touch $lockfile_run

                if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                    echo "Preprocessing $dataset with $n_view views"
                    /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                fi

                echo "Running $dataset with $n_view views with L2 loss and albedo"
                if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                    if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-50000/norm2/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-50000_norm2_woalbedo_exp${exp}.obj" ]; then
                        /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-50000/norm2/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-50000_norm2_woalbedo_exp${exp}.obj"
                    elif [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj" ]; then
                        /mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_50000_.obj"
                    fi
                fi

                ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter 50000 --ltwo --supernormal --rgbplus

                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path1
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path1
                mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_* $output_path1/mesh_50000_.obj

                rm -f $lockfile_run
            fi

            if [ -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                echo "Cleaning up $dataset with $n_view views"
                rm -rf "$data_dir_view/$dataset/rnbneus_data"
            fi

            if [ -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                echo "Cleaning up $dataset with $n_view views"
                rm -rf "$data_dir_view/$dataset/rnbneus_data-albedoscaled"
            fi

            # Stop 
            # exit 0

        done
    done
done
