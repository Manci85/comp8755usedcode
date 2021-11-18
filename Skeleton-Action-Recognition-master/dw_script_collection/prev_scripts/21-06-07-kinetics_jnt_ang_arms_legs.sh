#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/kinetics-skeleton/ang_enc/train_jnt_1ht_only_sgcn.yaml \
     >> outs_files/21-06-11-kinetics_jnt_1ht_only_sgcn.log 2>&1 &
