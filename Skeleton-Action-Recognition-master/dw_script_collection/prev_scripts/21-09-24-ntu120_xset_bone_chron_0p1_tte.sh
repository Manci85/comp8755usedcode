#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/2021-qin-msgcn/ntu120-xset/bone/train_chron_tte.yaml \
     >> outs_files/21-09-24-ntu120_xset_bone_chron_0p1_tte.log 2>&1 &
