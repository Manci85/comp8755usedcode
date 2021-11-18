#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config ./config/model_evaluation/ntu60_xsub/2019_icip_ra_gcn/train.yaml \
     >> outs_files/21-09-09-ntu_xsub_2019_icip_ra_gcn.log 2>&1 &

