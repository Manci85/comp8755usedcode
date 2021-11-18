#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config ./config/nturgbd120-cross-subject/angle_analysis/train_jnt_center_only_sgcn.yaml \
     >> outs_files/21-05-08-ntu120_xsub_jnt_center_1ht_only_sgcn.log 2>&1 &
