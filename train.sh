#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
--log_dir /home/yusongli/Documents/3D-UCaps/logs \
--gpus 1 \
--accelerator ddp \
--check_val_every_n_epoch 1 \
--max_epochs 1 \
--dataset yusongli \
--model_name ucaps \
--root_dir /root \
--fold 0 \
--cache_rate 1.0 \
--train_patch_size 32 32 32 \
--num_workers 64 \
--batch_size 4 \
--share_weight 0 \
--num_samples 4 \
--in_channels 1 \
--out_channels 3 \
--val_patch_size 32 32 32 \
--val_frequency 1 \
--sw_batch_size 16 \
--overlap 0.75
