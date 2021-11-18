#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config ./config/anubis/train_anubis_shift_gcn.yaml \
     >> outs_files/21-08-31-anubis_shift_gcn.log 2>&1 &
