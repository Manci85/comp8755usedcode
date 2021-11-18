#!/usr/bin/env bash

rsync -e 'ssh -p 22' \
    -av --update b:"/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/train_trig_temp_enc_jnt_8_w_orig.npy" \
    "/media/zhenyue-qin/Backup Plus/Transferring-Data" \
    --info=progress2
