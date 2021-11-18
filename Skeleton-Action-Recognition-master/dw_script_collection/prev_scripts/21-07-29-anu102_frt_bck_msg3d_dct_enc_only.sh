#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config ./config/anu101/train_anu101_msg3d_dct_enc_only.yaml \
     >> outs_files/21-07-29-train_anu101_msg3d_dct_enc_only.log 2>&1 &

