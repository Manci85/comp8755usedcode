#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python test_main.py \
    --config ./config/test_site/frame_idx_regression.yaml \
     >> outs_files/21-09-13-ntu_xsub_frame_idx_regression.log 2>&1 &
