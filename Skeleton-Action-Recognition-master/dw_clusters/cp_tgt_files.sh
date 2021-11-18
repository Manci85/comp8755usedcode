#!/usr/bin/env bash

# 从本地copy到服务器
#rsync -e 'ssh -p 22' \
#    -av --update \
#    "/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/train_data_joint.npy" \
#    "s:/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/"
#
#rsync -e 'ssh -p 22' \
#    -av --update \
#    "/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_data_joint.npy" \
#    "s:/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/"
#
#rsync -e 'ssh -p 22' \
#    -av --update \
#    "/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/train_data_joint.npy" \
#    "s:/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/"
#
#rsync -e 'ssh -p 22' \
#    -av --update \
#    "/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/val_data_joint.npy" \
#    "s:/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/"

rsync  -e 'ssh -p 22' \
    -av --update "/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xview/" \
    /data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/ \
    --info=progress2

rsync  -e 'ssh -p 22' \
    -av --update "/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/" \
    /data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/ \
    --info=progress2

rsync  -e 'ssh -p 22' \
    -av --update "/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/" \
    /data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/ \
    --info=progress2

rsync  -e 'ssh -p 22' \
    -av --update "/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/" \
    /data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/ \
    --info=progress2

#rsync  -e 'ssh -p 22' \
#    -av --update "/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/train_label.pkl" \
#    /data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/train_label.pkl \
#    --info=progress2

#rsync  -e 'ssh -p 22' \
#    -av --update "/data/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_label.pkl" \
#    /data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_label.pkl \
#    --info=progress2
