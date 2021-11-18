#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config config/2021-qin-msgcn/ntu60-xview/joint/train_chron_tte.yaml \
     >> outs_files/21-09-25-ntu_xview_joint_chron_0p1_tte.log 2>&1 &
