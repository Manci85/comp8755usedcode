#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py \
    --config ./config/model_evaluation/ntu60_xsub/2019_aaai_motif_stgcn/train_tte.yaml \
     >> outs_files/21-09-09-ntu_xsub_2019_aaai_motif_st_gcn_tte.log 2>&1 &

