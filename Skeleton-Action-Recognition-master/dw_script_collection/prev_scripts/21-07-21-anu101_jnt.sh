#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --config ./config/anu101/train_anu101.yaml \
     >> outs_files/21-07-21-train_anu101.log 2>&1 &
