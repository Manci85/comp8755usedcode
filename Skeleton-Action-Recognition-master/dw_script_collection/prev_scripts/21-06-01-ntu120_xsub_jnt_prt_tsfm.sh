#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config ./config/nturgbd120-cross-subject/pretrained_transformer/train_jnt_1ht_prtrn_tsfm.yaml \
     >> outs_files/21-06-07-ntu120_xsub_jnt_prt_tsfm.log 2>&1 &
