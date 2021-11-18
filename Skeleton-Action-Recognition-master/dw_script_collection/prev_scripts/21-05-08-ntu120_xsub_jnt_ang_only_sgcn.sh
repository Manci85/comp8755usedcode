#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2 python main.py \
    --config ./config/nturgbd120-cross-subject/angle_analysis/train_jnt_ang_only_sgcn.yaml \
     >> outs_files/21-05-08-ntu120_xsub_jnt_ang_only_sgcn.log 2>&1 &
