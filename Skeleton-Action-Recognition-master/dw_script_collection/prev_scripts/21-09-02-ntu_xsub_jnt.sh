#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config ./config/nturgbd-cross-subject/baseline/trn_jnt.yaml \
     >> outs_files/21-09-02-ntu_xsub_jnt.log 2>&1 &
