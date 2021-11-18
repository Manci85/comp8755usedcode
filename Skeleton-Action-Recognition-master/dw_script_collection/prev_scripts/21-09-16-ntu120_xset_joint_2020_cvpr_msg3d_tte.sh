#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/2020-cvpr-msg3d/ntu120-xset/joint/train_tte.yaml \
     >> outs_files/21-09-16-ntu120_xset_joint_2020_cvpr_msg3d_tte.log 2>&1 &
