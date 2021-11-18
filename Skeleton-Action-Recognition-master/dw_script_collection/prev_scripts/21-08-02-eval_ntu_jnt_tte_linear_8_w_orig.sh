#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config ./config/nturgbd-cross-subject/eval_scripts/val_jnt_tte_linear_8_w_orig.yaml \
     >> outs_files/21-08-02-val_ntu_jnt_tte_linear_8_w_orig.log 2>&1 &
