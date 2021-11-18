#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/kinetics-skeleton/ang_enc/train_jnt_bon_ang_arms_legs_v_1ht_only_sgcn.yaml \
     >> outs_files/21-06-06-kinetics_jnt_bon_ang_arms_legs_v_1ht_only_sgcn.log 2>&1 &
