#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/2021-qin-msgcn/ntu60-xsub/joint/train.yaml \
     >> outs_files/21-09-21-ntu_xsub_joint_qin_msgcn.log 2>&1 &
