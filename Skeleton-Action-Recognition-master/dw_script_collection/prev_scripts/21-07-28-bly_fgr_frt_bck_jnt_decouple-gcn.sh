#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main_bly_fgr.py \
    --config ./config/byl-fgr/train_byl_fgr_frt_bck_decouple-gcn.yaml \
     >> outs_files/21-07-28-bly_fgr_frt_bck_decouple-gcn.log 2>&1 &
