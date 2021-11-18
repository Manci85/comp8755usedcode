#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/cvpr_2020_msg3d/ntu60-xsub/joint/train_chron.yaml \
     >> outs_files/21-10-08-ntu60_xsub_cvpr_2020_msg3d_chron.log 2>&1 &
