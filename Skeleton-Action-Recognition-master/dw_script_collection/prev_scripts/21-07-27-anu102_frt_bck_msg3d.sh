#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config ./config/anu101/train_anu101_msg3d.yaml \
     >> outs_files/21-07-27-train_anu101_msg3d.log 2>&1 &

