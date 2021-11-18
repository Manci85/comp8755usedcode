#!/usr/bin/env bash

python main.py \
    --config ./config/nturgbd120-cross-subject/encode/train_jnt_dct_pos_only_sgcn.yaml \
     >> outs_files/21-04-08-ntu120_xsub_jnt_dct_pos_2_sin_cos_only_sgcn.log 2>&1 &
