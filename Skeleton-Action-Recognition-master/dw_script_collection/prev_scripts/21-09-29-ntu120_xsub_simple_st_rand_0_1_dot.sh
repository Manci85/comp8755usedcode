#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config config/qin_2021_simple_st/ntu120_xsub/train_rand_0_1_dot.yaml \
     >> outs_files/21-09-29-ntu120_xsub_simple_st_rand_0_1_dot.log 2>&1 &
