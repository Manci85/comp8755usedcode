#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/nturgbd120-cross-subject/siren/train_jnt_only_sgcn.yaml \
     >> outs_files/21-06-14-ntu120_xsub_jnt_1ht.log 2>&1 &
