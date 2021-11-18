#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config ./config/model_evaluation/ntu60_xsub/2019_cvpr_dual_agcn/train.yaml \
     >> outs_files/21-09-07-ntu_xsub_2019_cvpr_dual_agcn.log 2>&1 &

