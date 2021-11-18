#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main_decouple_gcn.py \
    --config config/eccv_2020_decouple_gcn/ntu60-xsub/train_chron.yaml \
     >> outs_files/21-10-08-ntu60_xsub_eccv_2020_decouple_gcn_chron.log 2>&1 &
