#!/usr/bin/env bash

rsync -av --update liu162@bracewell.hpc.csiro.au:/flush5/liu162/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/*.out \
    "/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_csiro/results_out"