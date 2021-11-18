#!/usr/bin/env bash

python main.py \
    --config ./config/nturgbd120-cross-subject/angle_analysis/train_jnt_unfixed_axis_only_sgcn.yaml \
     >> outs_files/21-05-04-ntu120_xsub_jnt_unfixed_axis_only_sgcn.log 2>&1 &
