#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/nturgbd-cross-subject/ste_k_analysis/trn_jnt_ste_3_linear_cos_w_orig.yaml \
     >> outs_files/21-08-04-ntu_jnt_ste_3_cos_linear_w_orig.log 2>&1 &
