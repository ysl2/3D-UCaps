#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python evaluate.py \
--root_dir /home/yusongli/_dataset/shidaoai/img/_out/nn/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/Task607_CZ2/labelsTr \
--output_dir /home/yusongli/_dataset/shidaoai/img/_out/nn/DATASET/nnUNet_cropped_data/nnUNet/3d_fullres/Task607_CZ2/3d-ucaps \
--gpus 1 \
--save_image 1 \
--model_name ucaps \
--dataset yusongli \
--fold 0 \
--checkpoint_path /home/yusongli/Documents/3D-UCaps/logs/ucaps_yusongli_0/version_0/checkpoints/epoch=0-val_dice=0.0177.ckpt \
--val_patch_size 32 32 32 \
--sw_batch_size 8 \
--overlap 0.75
