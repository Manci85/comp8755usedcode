#!/usr/bin/env bash

rsync -av --update "/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/" \
    liu162@bracewell.hpc.csiro.au:/flush5/liu162/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ \
    --info=progress2 \
    --exclude 'work_dir' \
    --exclude 'out_files' \
    --exclude 'results_csiro' \
    --exclude 'Exp_Storage' \
    --exclude 'Exp_Uncompleted' \
    --exclude 'pretrain_eval' \
    --exclude 'Motion-Data' \
    --exclude 'data' \
    --exclude 'results_dw_beijing' \
    --exclude 'eval' \
    --exclude 'ensemle_collection' \
    --exclude 'ensemble_collection'