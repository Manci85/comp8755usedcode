#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config config/qin_2021_simple_st/train.yaml \
     >> outs_files/21-09-28-ntu120_xsub_simple_st.log 2>&1 &
