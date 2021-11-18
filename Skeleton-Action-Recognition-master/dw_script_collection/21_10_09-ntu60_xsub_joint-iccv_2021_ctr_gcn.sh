#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config config/iccv_2021_ctr_gcn/ntu60-xsub/joint/train.yaml \
     >> outs_files/21_10_09-ntu60_xsub_joint-iccv_2021_ctr_gcn.log 2>&1 &
