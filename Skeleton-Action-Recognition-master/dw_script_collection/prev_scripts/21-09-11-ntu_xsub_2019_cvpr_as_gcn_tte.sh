#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_as_gcn.py \
    recognition -c config/model_evaluation/ntu60_xsub/2019_cvpr_as_gcn/train_tte.yaml \
    >> outs_files/21-09-11-ntu_xsub_2019_cvpr_as_gcn_tte.log 2>&1 &
