#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/nturgbd120-cross-setup/table_filling/train_jnt_bon_1ht_only_sgcn.yaml \
     >> outs_files/21-05-27-ntu120_xset_jnt_bon_1ht_only_sgcn.log 2>&1 &
