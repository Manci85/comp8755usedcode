#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/nturgbd-cross-subject/get_pretrain_model/train_jnt_bon_ang_loc_cen_hand_fing_vel_1ht_only_sgcn.yaml \
     >> outs_files/21-05-30-train_jnt_bon_ang_loc_cen_hand_fing_vel_1ht_only_sgcn.log 2>&1 &
