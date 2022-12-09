#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
--log_dir /home/yusongli/Documents/3D-UCaps/logs \
--dataset yusongli \
--gpus 1 \
--batch_size 4 \
--num_samples 1 \
--max_epochs 500 \
--in_channels 1 \
--out_channels 2 \
--train_patch_size 24 64 80 \
--val_patch_size 24 64 80 \
\
--accelerator ddp \
--check_val_every_n_epoch 1 \
--model_name ucaps \
--root_dir /root \
--fold 0 \
--cache_rate 1.0 \
--num_workers 64 \
--share_weight 0 \
--val_frequency 1 \
--sw_batch_size 16 \
--overlap 0.75

# --train_patch_size 32 32 32 \
# --val_patch_size 32 32 32 \
