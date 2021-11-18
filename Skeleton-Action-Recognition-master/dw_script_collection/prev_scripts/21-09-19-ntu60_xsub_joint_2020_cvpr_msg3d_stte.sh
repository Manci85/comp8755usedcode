#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/2021-qin-msgcn/ntu60-xsub/joint/train_stte.yaml \
     >> outs_files/21-09-19-ntu60_xsub_joint_2021_qin_msgcn_stte.log 2>&1 &
