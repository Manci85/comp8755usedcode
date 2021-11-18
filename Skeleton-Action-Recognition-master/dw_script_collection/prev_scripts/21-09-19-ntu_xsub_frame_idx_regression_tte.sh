#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python test_main.py \
    --config ./config/2021-qin-msgcn/ntu60-xsub/chronological_order/frame_idx_regression_tte_k_8.yaml \
     >> outs_files/21-09-19-ntu_xsub_frame_idx_regression_tte_k_8.log 2>&1 &
