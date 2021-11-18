#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config ./config/anubis/frt-end/2020_eccv_decouple_gcn/train_tte.yaml \
     >> outs_files/21-09-13-anubis_frt_bck_2020_eccv_decouple_gcn_tte.log 2>&1 &
