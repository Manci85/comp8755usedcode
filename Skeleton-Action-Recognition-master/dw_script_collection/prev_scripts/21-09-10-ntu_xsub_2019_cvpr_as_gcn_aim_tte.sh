#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python main_as_gcn.py \
    recognition -c config/model_evaluation/ntu60_xsub/2019_cvpr_as_gcn/train_aim_tte.yaml \
    >> outs_files/21-09-10-ntu_xsub_2019_cvpr_as_gcn_aim_tte.log 2>&1 &
