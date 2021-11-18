#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/nturgbd-cross-subject/dct/train_jnt_dct_3_only-enc.yaml \
     >> outs_files/21-07-27-jnt_dct_3_only-enc.log 2>&1 &
