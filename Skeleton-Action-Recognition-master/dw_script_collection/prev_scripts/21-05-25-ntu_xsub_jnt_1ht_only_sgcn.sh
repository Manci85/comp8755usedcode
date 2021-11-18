#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/nturgbd-cross-subject/table_filling/train_jnt_1ht_only_sgcn.yaml \
     >> outs_files/21-05-25-ntu120_xsub_jnt_only_sgcn.log 2>&1 &
