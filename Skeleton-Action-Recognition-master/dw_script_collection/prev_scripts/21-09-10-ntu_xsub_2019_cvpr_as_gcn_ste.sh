#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main_as_gcn.py \
    recognition -c config/model_evaluation/ntu60_xsub/2019_cvpr_as_gcn/train_ste.yaml \
    >> outs_files/21-09-10-ntu_xsub_2019_cvpr_as_gcn_ste.log 2>&1 &
