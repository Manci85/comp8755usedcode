#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2 python main.py \
    --config ./config/nturgbd-cross-subject/table_filling/train_bon_ang_all_vel_1ht_only_sgcn.yaml \
     >> outs_files/21-05-26-ntu120_xsub_bon_ang_all_vel_1ht_only_sgcn.log 2>&1 &
