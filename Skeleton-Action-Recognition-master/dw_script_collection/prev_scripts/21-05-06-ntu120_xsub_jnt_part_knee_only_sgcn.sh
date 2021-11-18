#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config ./config/nturgbd120-cross-subject/angle_analysis/train_jnt_part_knee_only_sgcn.yaml \
     >> outs_files/21-05-06-ntu120_xsub_jnt_part_knee_only_sgcn.log 2>&1 &
