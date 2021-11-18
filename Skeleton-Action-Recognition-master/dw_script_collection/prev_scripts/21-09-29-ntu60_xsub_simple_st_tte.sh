#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python main.py \
    --config config/qin_2021_simple_st/ntu60_xsub/train_tte.yaml \
     >> outs_files/21-09-29-ntu60_xsub_simple_st_tte.log 2>&1 &
