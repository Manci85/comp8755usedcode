#!/usr/bin/env bash
sleep 14400

python main.py \
    --config ./config/nturgbd120-cross-subject/transformer/train_bone_velocity_only_sgcn_temporal_transformer.yaml \
     >> mikoto_logs/20-12-05-ntu120_xsub_bone_velocity_only_sgcn_temp_trans_end_tcn_section_60_30_15.log 2>&1 &
