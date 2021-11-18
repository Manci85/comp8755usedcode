#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python test_main.py \
    --config ./config/test_site/frame_idx_regression_ste_k_8.yaml \
     >> outs_files/21-09-03-ntu_xsub_frame_idx_regression_ste_k_8.log 2>&1 &
