#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config config/2021-qin-msgcn/ntu120-xsub/train_tte.yaml \
     >> outs_files/21-09-15-ntu120_xsub_2021_qin_msgcn_tte_k_3.log 2>&1 &
