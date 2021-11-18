#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_bly_fgr.py \
    --config ./config/byl-fgr/train_byl_fgr.yaml \
     >> outs_files/21-07-16-bly_fgr_jnt.log 2>&1 &
