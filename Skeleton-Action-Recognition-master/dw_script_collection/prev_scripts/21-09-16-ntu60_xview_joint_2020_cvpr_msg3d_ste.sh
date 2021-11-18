#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/2020-cvpr-msg3d/ntu60-xview/joint/train_ste.yaml \
     >> outs_files/21-09-16-ntu60_xview_joint_2020_cvpr_msg3d_ste.log 2>&1 &
