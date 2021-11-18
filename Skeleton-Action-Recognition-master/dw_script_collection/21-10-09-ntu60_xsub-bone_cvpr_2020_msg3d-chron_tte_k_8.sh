#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --config config/cvpr_2020_msg3d/ntu60-xsub/bone/train_chron_tte.yaml \
     >> outs_files/21-10-09_ntu60-xsub-bone_2020_cvpr_msg3d-chron_tte_k_8.log 2>&1 &
