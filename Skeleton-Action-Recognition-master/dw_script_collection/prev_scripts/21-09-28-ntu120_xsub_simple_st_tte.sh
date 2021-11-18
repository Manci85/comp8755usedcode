#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 python main.py \
    --config config/qin_2021_simple_st/train_tte.yaml \
     >> outs_files/21-09-28-ntu120_xsub_simple_st_tte.log 2>&1 &
