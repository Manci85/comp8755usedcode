#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config config/qin_2021_component_lab/ntu120_xsub/joint/train.yaml \
     >> outs_files/21_10_12-ntu120_xsub_joint-qin_2021_lab-msgcn_T_600.log 2>&1 &
