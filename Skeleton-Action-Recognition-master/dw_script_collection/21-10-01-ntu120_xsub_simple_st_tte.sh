#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config config/qin_2021_simple_st/ntu120_xsub/train_tte.yaml \
     >> outs_files/21-10-01-ntu120_xsub_simple_st_tte.log 2>&1 &
