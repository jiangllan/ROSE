#!/bin/bash

model_dir=$1
task=$2

export CUDA_DEVICE_ORDER=PCI_BUS_ID

python ../run_classification.py \
   --model_name_or_path ${model_dir} \
   --do_eval \
   --output_dir ${model_dir} \
   --task_name ${task} \
   --overwrite_output_dir \
   --per_device_eval_batch_size 32 \
   --evaluation_strategy steps \
   --max_seq_length 128 \
   --fp16