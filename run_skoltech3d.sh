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

touch $lockfile_rnb
trap "rm -f $lockfile_rnb" EXIT

# PYTHON_PATH="/home/bbrument/anaconda3/envs/rnbneus2/bin"
PYTHON_PATH="/mnt/sdc1/bbrument/anaconda3/envs/rnbneus2/bin"

data_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/sdmunips"
results_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/results/sdmunips"
eval_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval"

exps=1
n_views="100"
# datasets="wooden_trex white_castle_land skate pink_boot orange_mini_vacuum white_human_skull green_bucket amber_vase small_wooden_chessboard golden_bust blue_boxing_gloves green_tea_boxes painted_samovar red_ceramic_fish painted_cup moon_pillow green_carved_pot jin_chan plush_bear golden_snail dragon"
datasets="dragon golden_snail plush_bear jin_chan green_carved_pot moon_pillow painted_cup red_ceramic_fish painted_samovar green_tea_boxes blue_boxing_gloves golden_bust small_wooden_chessboard amber_vase green_bucket white_human_skull orange_mini_vacuum pink_boot skate white_castle_land wooden_trex"
option_loss="norm1" # norm2"
n_iters="50000"

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

            if [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/worgbplus/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/worgbplus/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/wrgbplus/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/wrgbplus/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ]; then
                continue
            fi

            if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm1/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm1_woalbedo_exp${exp}.obj" ]; then
                if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm1/walbedo/worgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm1_walbedo_worgbplus_exp${exp}.obj" ]; then
                    if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm1/walbedo/wrgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm1_walbedo_wrgbplus_exp${exp}.obj" ]; then
                        if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm2/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm2_woalbedo_exp${exp}.obj" ]; then
                            if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm2/walbedo/worgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm2_walbedo_worgbplus_exp${exp}.obj" ]; then
                                if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm2/walbedo/wrgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm2_walbedo_wrgbplus_exp${exp}.obj" ]; then
                                    continue
                                fi
                            fi
                        fi
                    fi
                fi
            fi

            if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm1/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm1_woalbedo_exp${exp}_cleaned.ply" ]; then
                if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm1/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm1_walbedo_worgbplus_exp${exp}_cleaned.ply" ]; then
                    if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm1/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm1_walbedo_wrgbplus_exp${exp}_cleaned.ply" ]; then
                        if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm2/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm2_woalbedo_exp${exp}_cleaned.ply" ]; then
                            if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm2/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm2_walbedo_worgbplus_exp${exp}_cleaned.ply" ]; then
                                if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/norm2/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_norm2_walbedo_wrgbplus_exp${exp}_cleaned.ply" ]; then
                                    continue
                                fi
                            fi
                        fi
                    fi
                fi
            fi

            if [ $n_view == "100" ]; then
                data_dir_view=$data_dir
            else
                data_dir_view="$data_dir-$n_view"

                if [ $n_view == "50" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98
                elif [ $n_view == "20" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 5 10 16 21 26 32 37 42 47 53 58 63 68 74 79 84 89 95 99
                elif [ $n_view == "10" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 11 22 32 44 55 66 77 88 99
                elif [ $n_view == "5" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 25 50 75 99
                elif [ $n_view == "2" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 99
                fi
            fi

            for option in $option_loss
            do

                echo "Processing $dataset with $n_view views for ${option}"

                output_path1="$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp"
                output_path2="$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj"
                output_path3="$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply"
                if [ ! -f "$output_path1/mesh_${n_iters}_.obj" ] && [ ! -f "$output_path2" ] && [ ! -f "$output_path3" ]; then

                    lockfile_run="$output_path1/lockfile_run"
                    if [ -f "$lockfile_run" ]; then
                        echo "Already running $dataset with $n_view views with ${option} loss and no albedo"
                        continue
                    fi
                    mkdir -p $output_path1
                    touch $lockfile_run

                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                        echo "Preprocessing $dataset with $n_view views"
                        $PYTHON_PATH/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                    fi

                    echo "Running $dataset with $n_view views with ${option} loss and no albedo"
                    if [ $option == "norm1" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter ${n_iters} --supernormal
                    elif [ $option == "norm2" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter ${n_iters} --ltwo --supernormal
                    fi
                    
                    mv $data_dir_view/$dataset/rnbneus_data/command.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data/run.sh.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data/mesh_* $output_path1/mesh_${n_iters}_.obj

                    rm -f $lockfile_run
                fi

                output_path1="$results_dir/$dataset/supernormal/wosnap2center/${option}/walbedo/worgbplus/n_views-$n_view/exp$exp"
                output_path2="$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/walbedo/worgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_walbedo_worgbplus_exp${exp}.obj"
                output_path3="$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_walbedo_worgbplus_exp${exp}_cleaned.ply"
                if [ ! -f "$output_path1/mesh_${n_iters}_.obj" ] && [ ! -f "$output_path2" ] && [ ! -f "$output_path3" ]; then

                    lockfile_run="$output_path1/lockfile_run"
                    if [ -f "$lockfile_run" ]; then
                        echo "Already running $dataset with $n_view views with ${option} loss and albedo"
                        continue
                    fi
                    mkdir -p $output_path1
                    touch $lockfile_run

                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                        echo "Preprocessing $dataset with $n_view views"
                        $PYTHON_PATH/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                    fi

                    echo "Running $dataset with $n_view views with ${option} loss and albedo"
                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                        if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj"
                        elif [ -f "$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj"
                        elif [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply"
                        fi
                    fi

                    if [ $option == "norm1" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter ${n_iters} --supernormal
                    elif [ $option == "norm2" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter ${n_iters} --ltwo --supernormal
                    fi

                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_* $output_path1/mesh_${n_iters}_.obj

                    rm -f $lockfile_run
                fi

                output_path1="$results_dir/$dataset/supernormal/wosnap2center/${option}/walbedo/wrgbplus/n_views-$n_view/exp$exp"
                output_path2="$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/walbedo/wrgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_walbedo_wrgbplus_exp${exp}.obj"
                output_path3="$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_walbedo_wrgbplus_exp${exp}_cleaned.ply"
                if [ ! -f "$output_path1/mesh_${n_iters}_.obj" ] && [ ! -f "$output_path2" ] && [ ! -f "$output_path3" ]; then

                    lockfile_run="$output_path1/lockfile_run"
                    if [ -f "$lockfile_run" ]; then
                        echo "Already running $dataset with $n_view views with ${option} loss and albedo and rgb+"
                        continue
                    fi
                    mkdir -p $output_path1
                    touch $lockfile_run

                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                        echo "Preprocessing $dataset with $n_view views"
                        $PYTHON_PATH/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                    fi

                    echo "Running $dataset with $n_view views with ${option} loss and albedo"
                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                        if [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj"
                        elif [ -f "$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj"
                        elif [ -f "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/sdmunips/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_sdmunips_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply"
                        fi
                    fi

                    if [ $option == "norm1" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter ${n_iters} --supernormal --rgbplus
                    elif [ $option == "norm2" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter ${n_iters} --ltwo --supernormal --rgbplus
                    fi

                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_* $output_path1/mesh_${n_iters}_.obj

                    rm -f $lockfile_run
                fi
            
            done

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


data_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/unimsps-16bits"
results_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/results/unimsps-16bits"
eval_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval"

exps=1
n_views="100 50 20 10 5 2"
# datasets="wooden_trex white_castle_land skate pink_boot orange_mini_vacuum white_human_skull green_bucket amber_vase small_wooden_chessboard golden_bust blue_boxing_gloves green_tea_boxes painted_samovar red_ceramic_fish painted_cup moon_pillow green_carved_pot jin_chan plush_bear golden_snail dragon"
datasets="dragon golden_snail plush_bear jin_chan green_carved_pot moon_pillow painted_cup red_ceramic_fish painted_samovar green_tea_boxes blue_boxing_gloves golden_bust small_wooden_chessboard amber_vase green_bucket white_human_skull orange_mini_vacuum pink_boot skate white_castle_land wooden_trex"
option_loss="norm1" # norm2"
n_iters="50000"

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

            if [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ]; then
                continue
            fi

            if [ -f "$eval_dir/$dataset/rnbv2-sn/unimsps-16bits/nbv-$n_view/nbit-${n_iters}/norm1/exp$exp/results_raw/${dataset}_rnbv2-sn_unimsps-16bits_nbv-${n_view}_nbit-${n_iters}_norm1_exp${exp}.obj" ] && [ -f "$eval_dir/$dataset/rnbv2-sn/unimsps-16bits/nbv-$n_view/nbit-${n_iters}/norm2/exp$exp/results_raw/${dataset}_rnbv2-sn_unimsps-16bits_nbv-${n_view}_nbit-${n_iters}_norm2_exp${exp}.obj" ]; then
                continue
            fi

            if [ -f "$eval_dir/$dataset/rnbv2-sn/unimsps-16bits/nbv-$n_view/nbit-${n_iters}/norm1/exp$exp/results_cleaned/${dataset}_rnbv2-sn_unimsps-16bits_nbv-${n_view}_nbit-${n_iters}_norm1_exp${exp}_cleaned.ply" ] && [ -f "$eval_dir/$dataset/rnbv2-sn/unimsps-16bits/nbv-$n_view/nbit-${n_iters}/norm2/exp$exp/results_cleaned/${dataset}_rnbv2-sn_unimsps-16bits_nbv-${n_view}_nbit-${n_iters}_norm2_exp${exp}_cleaned.ply" ]; then
                continue
            fi

            if [ $n_view == "100" ]; then
                data_dir_view=$data_dir
            else
                data_dir_view="$data_dir-$n_view"

                if [ $n_view == "50" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98
                elif [ $n_view == "20" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 5 10 16 21 26 32 37 42 47 53 58 63 68 74 79 84 89 95 99
                elif [ $n_view == "10" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 11 22 33 44 55 66 77 88 99
                elif [ $n_view == "5" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 25 50 75 99
                elif [ $n_view == "2" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 99
                fi
            fi

            for option in $option_loss
            do

                output_path1="$results_dir/$dataset/supernormal/wosnap2center/${option}/n_views-$n_view/exp$exp"
                output_path2="$eval_dir/$dataset/rnbv2-sn/unimsps-16bits/nbv-$n_view/nbit-${n_iters}/${option}/exp$exp/results_raw/${dataset}_rnbv2-sn_unimsps-16bits_nbv-${n_view}_nbit-${n_iters}_${option}_exp${exp}.obj"
                output_path3="$eval_dir/$dataset/rnbv2-sn/unimsps-16bits/nbv-$n_view/nbit-${n_iters}/${option}/exp$exp/results_cleaned/${dataset}_rnbv2-sn_unimsps-16bits_nbv-${n_view}_nbit-${n_iters}_${option}_exp${exp}_cleaned.ply"
                if [ ! -f "$output_path1/mesh_${n_iters}_.obj" ] && [ ! -f "$output_path2" ] && [ ! -f "$output_path3" ]; then

                    lockfile_run="$output_path1/lockfile_run"
                    if [ -f "$lockfile_run" ]; then
                        echo "Already running $dataset with $n_view views with ${option} loss"
                        continue
                    fi
                    mkdir -p $output_path1
                    touch $lockfile_run

                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                        echo "Preprocessing $dataset with $n_view views"
                        $PYTHON_PATH/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                    fi

                    echo "Running $dataset with $n_view views with ${option} loss"
                    if [ $option == "norm1" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter ${n_iters} --supernormal
                    elif [ $option == "norm2" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter ${n_iters} --ltwo --supernormal
                    fi
                    
                    mv $data_dir_view/$dataset/rnbneus_data/command.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data/run.sh.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data/mesh_* $output_path1/mesh_${n_iters}_.obj

                    rm -f $lockfile_run
                fi

            done

            if [ -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                echo "Cleaning up $dataset with $n_view views"
                rm -rf "$data_dir_view/$dataset/rnbneus_data"
            fi
        done
    done
done



data_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/normals_gt"
results_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/results/normals_gt"
eval_dir="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/eval"

exps=1
n_views="100"
# datasets="wooden_trex white_castle_land skate pink_boot orange_mini_vacuum white_human_skull green_bucket amber_vase small_wooden_chessboard golden_bust blue_boxing_gloves green_tea_boxes painted_samovar red_ceramic_fish painted_cup moon_pillow green_carved_pot jin_chan plush_bear golden_snail dragon"
datasets="dragon golden_snail plush_bear jin_chan green_carved_pot moon_pillow painted_cup red_ceramic_fish painted_samovar green_tea_boxes blue_boxing_gloves golden_bust small_wooden_chessboard amber_vase green_bucket white_human_skull orange_mini_vacuum pink_boot skate white_castle_land wooden_trex"
option_loss="norm2 norm1"
n_iters="50000"

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

            if [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/worgbplus/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/worgbplus/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm1/walbedo/wrgbplus/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ] && [ -f "$results_dir/$dataset/supernormal/wosnap2center/norm2/walbedo/wrgbplus/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ]; then
                continue
            fi

            if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm1/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm1_woalbedo_exp${exp}.obj" ]; then
                if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm1/walbedo/worgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm1_walbedo_worgbplus_exp${exp}.obj" ]; then
                    if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm1/walbedo/wrgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm1_walbedo_wrgbplus_exp${exp}.obj" ]; then
                        if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm2/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm2_woalbedo_exp${exp}.obj" ]; then
                            if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm2/walbedo/worgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm2_walbedo_worgbplus_exp${exp}.obj" ]; then
                                if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm2/walbedo/wrgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm2_walbedo_wrgbplus_exp${exp}.obj" ]; then
                                    continue
                                fi
                            fi
                        fi
                    fi
                fi
            fi

            if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm1/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm1_woalbedo_exp${exp}_cleaned.ply" ]; then
                if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm1/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm1_walbedo_worgbplus_exp${exp}_cleaned.ply" ]; then
                    if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm1/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm1_walbedo_wrgbplus_exp${exp}_cleaned.ply" ]; then
                        if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm2/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm2_woalbedo_exp${exp}_cleaned.ply" ]; then
                            if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm2/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm2_walbedo_worgbplus_exp${exp}_cleaned.ply" ]; then
                                if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/norm2/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_norm2_walbedo_wrgbplus_exp${exp}_cleaned.ply" ]; then
                                    continue
                                fi
                            fi
                        fi
                    fi
                fi
            fi

            if [ $n_view == "100" ]; then
                data_dir_view=$data_dir
            else
                data_dir_view="$data_dir-$n_view"

                if [ $n_view == "50" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98
                elif [ $n_view == "20" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 5 10 16 21 26 32 37 42 47 53 58 63 68 74 79 84 89 95 99
                elif [ $n_view == "10" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 11 22 32 44 55 66 77 88 99
                elif [ $n_view == "5" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 25 50 75 99
                elif [ $n_view == "2" ]; then
                    $PYTHON_PATH/python select_ind_idr.py --data_path $data_dir/$dataset --output_path $data_dir_view/$dataset --ind_images 0 99
                fi
            fi

            for option in $option_loss
            do

                echo "Processing $dataset with $n_view views for ${option}"

                output_path1="$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp"
                output_path2="$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj"
                output_path3="$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply"
                if [ ! -f "$output_path1/mesh_${n_iters}_.obj" ] && [ ! -f "$output_path2" ] && [ ! -f "$output_path3" ]; then

                    lockfile_run="$output_path1/lockfile_run"
                    if [ -f "$lockfile_run" ]; then
                        echo "Already running $dataset with $n_view views with ${option} loss and no albedo"
                        continue
                    fi
                    mkdir -p $output_path1
                    touch $lockfile_run

                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                        echo "Preprocessing $dataset with $n_view views"
                        $PYTHON_PATH/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                    fi

                    echo "Running $dataset with $n_view views with ${option} loss and no albedo"
                    if [ $option == "norm1" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter ${n_iters} --supernormal
                    elif [ $option == "norm2" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data --no-albedo --res 768 --disable-snap-to-center --num-iter ${n_iters} --ltwo --supernormal
                    fi
                    
                    mv $data_dir_view/$dataset/rnbneus_data/command.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data/run.sh.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data/mesh_* $output_path1/mesh_${n_iters}_.obj

                    rm -f $lockfile_run
                fi

                output_path1="$results_dir/$dataset/supernormal/wosnap2center/${option}/walbedo/worgbplus/n_views-$n_view/exp$exp"
                output_path2="$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/walbedo/worgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_walbedo_worgbplus_exp${exp}.obj"
                output_path3="$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/walbedo/worgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_walbedo_worgbplus_exp${exp}_cleaned.ply"
                if [ ! -f "$output_path1/mesh_${n_iters}_.obj" ] && [ ! -f "$output_path2" ] && [ ! -f "$output_path3" ]; then

                    lockfile_run="$output_path1/lockfile_run"
                    if [ -f "$lockfile_run" ]; then
                        echo "Already running $dataset with $n_view views with ${option} loss and albedo"
                        continue
                    fi
                    mkdir -p $output_path1
                    touch $lockfile_run

                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                        echo "Preprocessing $dataset with $n_view views"
                        $PYTHON_PATH/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                    fi

                    echo "Running $dataset with $n_view views with ${option} loss and albedo"
                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                        if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj"
                        elif [ -f "$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj"
                        elif [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply"
                        fi
                    fi

                    if [ $option == "norm1" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter ${n_iters} --supernormal
                    elif [ $option == "norm2" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter ${n_iters} --ltwo --supernormal
                    fi

                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_* $output_path1/mesh_${n_iters}_.obj

                    rm -f $lockfile_run
                fi

                output_path1="$results_dir/$dataset/supernormal/wosnap2center/${option}/walbedo/wrgbplus/n_views-$n_view/exp$exp"
                output_path2="$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/walbedo/wrgbplus/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_walbedo_wrgbplus_exp${exp}.obj"
                output_path3="$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/walbedo/wrgbplus/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_walbedo_wrgbplus_exp${exp}_cleaned.ply"
                if [ ! -f "$output_path1/mesh_${n_iters}_.obj" ] && [ ! -f "$output_path2" ] && [ ! -f "$output_path3" ]; then

                    lockfile_run="$output_path1/lockfile_run"
                    if [ -f "$lockfile_run" ]; then
                        echo "Already running $dataset with $n_view views with ${option} loss and albedo and rgb+"
                        continue
                    fi
                    mkdir -p $output_path1
                    touch $lockfile_run

                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data/transform.json" ]; then
                        echo "Preprocessing $dataset with $n_view views"
                        $PYTHON_PATH/python scripts/preprocess.py --folder $data_dir_view/$dataset --exp_name "rnbneus_data" --mask_certainty_name "mask"
                    fi

                    echo "Running $dataset with $n_view views with ${option} loss and albedo"
                    if [ ! -f "$data_dir_view/$dataset/rnbneus_data-albedoscaled/transform.json" ]; then
                        if [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_raw/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}.obj"
                        elif [ -f "$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$results_dir/$dataset/supernormal/wosnap2center/${option}/woalbedo/n_views-$n_view/exp$exp/mesh_${n_iters}_.obj"
                        elif [ -f "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply" ]; then
                            $PYTHON_PATH/python scripts/scale_albedos.py --folder $data_dir_view/$dataset/rnbneus_data --mesh_path "$eval_dir/$dataset/rnbv2-sn/normals_gt/nbv-$n_view/nbit-${n_iters}/${option}/woalbedo/exp$exp/results_cleaned/${dataset}_rnbv2-sn_normals_gt_nbv-${n_view}_nbit-${n_iters}_${option}_woalbedo_exp${exp}_cleaned.ply"
                        fi
                    fi

                    if [ $option == "norm1" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter ${n_iters} --supernormal --rgbplus
                    elif [ $option == "norm2" ]; then
                        ./run.sh $data_dir_view/$dataset/rnbneus_data-albedoscaled --res 768 --disable-snap-to-center --num-iter ${n_iters} --ltwo --supernormal --rgbplus
                    fi

                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/command.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/run.sh.log $output_path1
                    mv $data_dir_view/$dataset/rnbneus_data-albedoscaled/mesh_* $output_path1/mesh_${n_iters}_.obj

                    rm -f $lockfile_run
                fi
            
            done

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