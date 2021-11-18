#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python main.py \
    --config ./config/model_evaluation/ntu60_xsub/2020_aaai_gcn_nas/train_tte.yaml \
     >> outs_files/21-09-08-ntu_xsub_2020_aaai_gcn_nas_tte.log 2>&1 &

