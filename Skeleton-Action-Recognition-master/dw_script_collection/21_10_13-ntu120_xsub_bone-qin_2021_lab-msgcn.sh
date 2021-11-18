#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/qin_2021_component_lab/ntu120_xsub/bone/train.yaml \
     >> outs_files/21_10_13-ntu120_xsub_bone-qin_2021_lab-msgcn.log 2>&1 &
