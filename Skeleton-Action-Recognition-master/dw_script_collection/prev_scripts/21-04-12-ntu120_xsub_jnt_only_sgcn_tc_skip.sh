#!/usr/bin/env bash

python main.py \
    --config ./config/nturgbd120-cross-subject/encode/train_jnt_temp_modify_only_sgcn.yaml \
     >> outs_files/21-04-12-ntu120_xsub_jnt_only_sgcn_tc_skip_cat.log 2>&1 &
