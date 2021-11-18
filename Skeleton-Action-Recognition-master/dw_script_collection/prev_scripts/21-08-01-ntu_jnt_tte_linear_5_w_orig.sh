#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/nturgbd-cross-subject/tte_k_analysis/trn_jnt_tte_linear_5_w_orig.yaml \
     >> outs_files/21-08-01-ntu_jnt_tte_linear_5_w_orig.log 2>&1 &
