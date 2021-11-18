#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config ./config/nturgbd-cross-subject/dct/train_jnt_dct_3_w-orig.yaml \
     >> outs_files/21-07-28-jnt_dct_3_w-orig.log 2>&1 &
