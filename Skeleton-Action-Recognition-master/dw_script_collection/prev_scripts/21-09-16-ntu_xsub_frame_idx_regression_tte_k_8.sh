#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python test_main.py \
    --config ./config/test_site/frame_idx_regression_tte_k_8.yaml \
     >> outs_files/21-09-16-ntu_xsub_frame_idx_regression_tte_k_8.log 2>&1 &
