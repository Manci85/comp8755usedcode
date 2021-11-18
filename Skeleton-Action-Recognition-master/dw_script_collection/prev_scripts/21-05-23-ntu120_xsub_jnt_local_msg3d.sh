#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 python main.py \
    --config ./config/nturgbd120-cross-subject/msg3d/train_jnt_ang_local_msg3d.yaml \
     >> outs_files/21-05-23-ntu120_xsub_jnt_ang_local_msg3d.log 2>&1 &
