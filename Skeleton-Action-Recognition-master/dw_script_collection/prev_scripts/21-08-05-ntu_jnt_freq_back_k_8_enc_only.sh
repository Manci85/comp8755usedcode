#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,3 python main.py \
    --config ./config/nturgbd-cross-subject/freq_back/trn_jnt_freq_back_k_8_enc_only.yaml \
     >> outs_files/21-08-05-ntu_jnt_freq_back_k_8_enc_only.log 2>&1 &
