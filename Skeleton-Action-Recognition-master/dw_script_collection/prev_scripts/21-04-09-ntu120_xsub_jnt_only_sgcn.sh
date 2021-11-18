#!/usr/bin/env bash

python main.py \
    --config ./config/nturgbd120-cross-subject/encode/train_jnt_only_sgcn.yaml \
     >> outs_files/21-04-09-ntu120_xsub_jnt_only_sgcn.log 2>&1 &
