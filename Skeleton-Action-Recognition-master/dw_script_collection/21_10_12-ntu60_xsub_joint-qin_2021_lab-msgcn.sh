#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config config/qin_2021_component_lab/ntu60_xsub/joint/train.yaml \
     >> outs_files/21_10_12-ntu60_xsub_joint-qin_2021_lab-msgcn.log 2>&1 &
