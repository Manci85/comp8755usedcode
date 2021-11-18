#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_dgnn.py \
    --config ./config/model_evaluation/ntu60_xsub/2019_cvpr_dgnn/train.yaml \
     >> outs_files/21-09-07-ntu_xsub_2019_cvpr_dgnn.log 2>&1 &

