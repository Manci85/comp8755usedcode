#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config config/qin_2021_simple_st/ntu120_xsub/train_repeat_data.yaml \
     >> outs_files/21-10-01-ntu120_xsub_simple_st_repeat_data.log 2>&1 &
