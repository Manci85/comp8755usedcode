#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 python main.py \
    --config config/2020-cvpr-msg3d/ntu60-xview/bone/train_chron_tte.yaml \
     >> outs_files/21-09-26-msg3d-ntu60_xview_bone_chron_0p1_tte.log 2>&1 &
