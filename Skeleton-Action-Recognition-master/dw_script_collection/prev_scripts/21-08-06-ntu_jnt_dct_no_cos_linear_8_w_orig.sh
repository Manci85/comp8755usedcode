#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/nturgbd-cross-subject/analyse_scaling/trn_jnt_dct_no_cos_linear_8_w_orig.yaml \
     >> outs_files/21-08-06-ntu_jnt_dct_no_cos_linear_8_w_orig.log 2>&1 &
