#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python test_main.py \
    --config ./config/test_site/frame_idx_regression_tte_k_8_rand.yaml \
     >> outs_files/21-09-12-ntu_xsub_frame_idx_regression_tte_k_8_rand.log 2>&1 &
