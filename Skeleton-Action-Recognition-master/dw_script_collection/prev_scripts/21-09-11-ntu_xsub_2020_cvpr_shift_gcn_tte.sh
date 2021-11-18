#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python main_shift_gcn.py \
    --config ./config/model_evaluation/ntu60_xsub/2020_cvpr_shift_gcn/train_tte.yaml \
     >> outs_files/21-09-11-ntu_xsub_2020_cvpr_shift_gcn_tte.log 2>&1 &
