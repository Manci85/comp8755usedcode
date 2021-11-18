#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py \
    --config ./config/model_evaluation/ntu60_xsub/2018_aaai_stgcn/train_ste.yaml \
     >> outs_files/21-09-08-ntu_xsub_2018_aaai_stgcn_ste.log 2>&1 &

