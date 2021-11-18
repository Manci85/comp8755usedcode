#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5 python main.py \
    --config ./config/nturgbd-cross-subject/enc_studies/trn_jnt_rand_dot_8_w_orig.yaml \
     >> outs_files/21-09-01-jnt_rand_dot_8_w_orig.log 2>&1 &
