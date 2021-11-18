#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config ./config/anubis/frt-end/2020_eccv_decouple_gcn/train_ste_enc_only.yaml \
     >> outs_files/21-09-13-anubis_frt_bck_2020_eccv_decouple_gcn_ste_enc_only.log 2>&1 &
