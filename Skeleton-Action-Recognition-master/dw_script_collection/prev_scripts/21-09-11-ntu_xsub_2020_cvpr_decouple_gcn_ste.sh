#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main_decouple_gcn.py \
    --config ./config/model_evaluation/ntu60_xsub/2020_cvpr_decouple_gcn/train_ste_orig.yaml \
     >> outs_files/21-09-11-ntu_xsub_2020_cvpr_decouple_gcn_ste.log 2>&1 &
