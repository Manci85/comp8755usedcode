#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/kinetics-skeleton/ang_enc/train_bon_ang_arms_legs_1ht_only_sgcn.yaml \
     >> outs_files/21-06-07-kinetics_bon_ang_arms_legs_1ht_only_sgcn.log 2>&1 &
