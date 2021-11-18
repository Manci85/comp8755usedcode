#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main_shift_gcn.py \
    --config ./config/anubis/frt-end/2020_cvpr_shift_gcn/train_ste.yaml \
     >> outs_files/21-09-12-anubis_frt_bck_2020_cvpr_shift_gcn_ste.log 2>&1 &
