#!/usr/bin/env bash

python main.py \
    --config ./config/nturgbd120-cross-subject/encode/train_jnt_dct_sum_1ht_only_sgcn.yaml \
     >> outs_files/21-04-07-ntu120_xsub_jnt_dct_sum_3_1ht_g3d.log 2>&1 &
