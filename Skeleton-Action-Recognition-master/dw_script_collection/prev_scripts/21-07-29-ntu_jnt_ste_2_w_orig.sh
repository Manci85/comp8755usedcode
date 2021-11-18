#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config ./config/nturgbd-cross-subject/ste_k_analysis/trn_jnt_ste_2_w_orig.yaml \
     >> outs_files/21-07-29-ntu_jnt_ste_2_w_orig.log 2>&1 &
