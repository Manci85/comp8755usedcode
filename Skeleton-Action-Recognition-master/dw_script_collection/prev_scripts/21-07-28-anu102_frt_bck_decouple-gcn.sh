#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config ./config/anu101/train_anu101_decouple-gcn.yaml \
     >> outs_files/21-07-28-train_anu101_decouple-gcn.log 2>&1 &

