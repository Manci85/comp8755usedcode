#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/nturgbd120-cross-subject/angle_analysis/train_bon_ang_only_sgcn.yaml \
     >> outs_files/21-05-08-ntu120_xsub_bon_ang_only_sgcn.log 2>&1 &
