#!/bin/bash

model_name=$1
n_GPU=$2
lr=$3
train_batch_size=$4
seed=$5
upper=$6
lower=$7
task=$8

max_seq_length=128
per_device_train_batch_size=${train_batch_size}
per_device_lr=${lr}
write_lr=$(echo "${lr} * 100000" | bc)
write_lr=$(echo "${write_lr}"|awk '{printf("%d",$0)}')
epoch=10

echo "learning_rate: $lr"
echo "epoch_num: $epoch"
echo "train_batch_size: $train_batch_size"


export CUDA_DEVICE_ORDER="PCI_BUS_ID"
python -m torch.distributed.launch \
 --nproc_per_node ${n_GPU} \
 --master_port $RANDOM../run_classification.py \
 --model_name_or_path ${model_name} \
 --do_eval \
 --output_dir ../temp/${model_name}/sparse_second/${task}_l${lower}_u${upper}/bs${train_batch_size}_lr${write_lr}_seed${seed} \
 --task_name ${task} \
 --overwrite_output_dir \
 --per_device_train_batch_size ${per_device_train_batch_size} \
 --per_device_eval_batch_size 32 \
 --num_train_epochs ${epoch} \
 --evaluation_strategy epoch \
 --logging_steps 200 \
 --metric_for_best_model accuracy \
 --greater_is_better yes \
 --learning_rate ${per_device_lr} \
 --max_seq_length ${max_seq_length} \
 --do_train \
 --fp16 \
 --overwrite_output_dir \
 --weight_decay 0.01 \
 --adam_beta2 0.98 \
 --do_sparse \
 --sparse_second \
 --upper ${upper} \
 --lower ${lower} \
 --seed ${seed}