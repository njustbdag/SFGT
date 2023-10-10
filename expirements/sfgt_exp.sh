#!/bin/bash

dataset_lists=("graph_YelpChi_heter.pk" "graph_Amazon_heter.pk")
data_dirs=("../data/yelpchi" "../data/amazon")
task_names=("yelpchi" "amazon")
hids=(400)
sample_widths=(128)
sample_depths=(6)
orders=(1 2 3 4 5 6)

for dataset in "${dataset_lists[@]}"; do
    for data_dir in "${data_dirs[@]}"; do
        for task_name in "${task_names[@]}"; do
            for hid in "${hids[@]}"; do
                for sample_depth in "${sample_depths[@]}"; do
                    for order in "${orders[@]}"; do
                        echo "$dataset"
                        python ../train/train.py --data_dir $data_dir --model_dir "../Models" --dataset $dataset --n_epoch 100 --n_hid $hid \
                        --task_name $task_name --dis_func "cos" --sample_depth $sample_depth --sample_width 16 --moments $order --conv_name "mmgt" \
                        --n_layers 2 --sample_threshold 1 \
                        --n_batch 12 --batch_size 256  --or_threshold 1000
                    done
                done
            done
        done
    done
done