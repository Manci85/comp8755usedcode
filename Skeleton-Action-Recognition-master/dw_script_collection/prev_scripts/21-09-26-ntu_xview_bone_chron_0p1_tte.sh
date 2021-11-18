#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/2021-qin-msgcn/ntu60-xview/bone/train_chron_tte.yaml \
     >> outs_files/21-09-26-ntu_xview_bone_chron_0p1_tte.log 2>&1 &
