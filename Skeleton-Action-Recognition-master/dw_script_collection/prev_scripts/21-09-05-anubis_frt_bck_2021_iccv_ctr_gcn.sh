#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2 python main.py \
    --config ./config/anubis/frt-end/dw/train_2021_iccv_ctr_gcn.yaml \
     >> outs_files/21-09-05-anubis_frt_bck_2021_iccv_ctr_gcn.log 2>&1 &
