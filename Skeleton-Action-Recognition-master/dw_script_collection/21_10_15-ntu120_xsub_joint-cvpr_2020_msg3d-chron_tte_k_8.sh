#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/cvpr_2020_msg3d/ntu120-xsub/joint/train_chron_tte.yaml \
     >> outs_files/21_10_15-ntu120_xsub_joint-cvpr_2020_msg3d-chron_tte_k_8.log 2>&1 &
