#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/cvpr_2020_msg3d/ntu120-xsub/joint/train_chron.yaml \
     >> outs_files/21-10-07-ntu120_xsub_cvpr_2020_msg3d_chron.log 2>&1 &
