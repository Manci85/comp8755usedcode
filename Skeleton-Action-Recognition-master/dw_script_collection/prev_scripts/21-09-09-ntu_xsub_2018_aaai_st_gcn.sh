#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config ./config/model_evaluation/ntu60_xsub/2018_aaai_stgcn/train.yaml \
     >> outs_files/21-09-09-ntu_xsub_2018_aaai_st_gcn.log 2>&1 &

