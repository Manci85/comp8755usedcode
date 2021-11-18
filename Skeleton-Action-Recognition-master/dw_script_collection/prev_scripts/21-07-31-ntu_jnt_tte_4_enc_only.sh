#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config ./config/nturgbd-cross-subject/tte_k_analysis/trn_jnt_tte_4_enc_only.yaml \
     >> outs_files/21-07-31-ntu_jnt_tte_4_enc_only.log 2>&1 &
