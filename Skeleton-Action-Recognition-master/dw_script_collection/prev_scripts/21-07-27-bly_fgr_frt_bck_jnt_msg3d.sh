#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main_bly_fgr.py \
    --config ./config/byl-fgr/train_byl_fgr_frt_bck_msg3d.yaml \
     >> outs_files/21-07-27-bly_fgr_frt_bck_msg3d.log 2>&1 &
