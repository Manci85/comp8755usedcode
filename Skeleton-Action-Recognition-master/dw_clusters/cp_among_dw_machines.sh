#!/usr/bin/env bash

scp -P 50022 -r "/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/train_data_joint_bone_angle_cangle_arms_legs.npy" \
    zhenyue@141.223.181.81:"/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/"

scp -P 50022 -r "/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/train_label.pkl" \
    zhenyue@141.223.181.81:"/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/"

scp -P 50022 -r "/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_data_joint_bone_angle_cangle_arms_legs.npy" \
    zhenyue@141.223.181.81:"/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/"

scp -P 50022 -r "/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_label.pkl" \
    zhenyue@141.223.181.81:"/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/"

#
#scp -P 50022 -r "/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/train_data_joint_bone_angle_cangle_arms_legs_velocity.npy" \
#    zhenyue@141.223.181.65:"/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/"
#
#scp -P 50022 -r "/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_data_joint_bone_angle_cangle_arms_legs_velocity.npy" \
#    zhenyue@141.223.181.65:"/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/"

#rsync  -e 'ssh -p 50022' \
#    -av --update "/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/" \
#    zhenyue@141.223.181.81:/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ \
#    --info=progress2 \
#    --exclude 'work_dir' \
#    --exclude 'out_files' \
#    --exclude 'results_csiro' \
#    --exclude 'data' \
#    --exclude 'pretrained-models' \
#    --exclude 'Exp_Storage' \
#    --exclude 'Exp_Uncompleted' \
#    --exclude 'pretrain_eval' \
#    --exclude 'Motion-Data' \
#    --exclude 'results_dw_beijing' \
#    --exclude 'eval' \
#    --exclude 'ensemble_collection' \
#    --exclude 'visuals/' \
#    --exclude 'test_fields' \
#    --exclude 'IJCAI-Results'

#rsync  -e 'ssh -p 50022' \
#    -av --update "/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/" \
#    zhenyue@141.223.181.82:/home/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ \
#    --info=progress2 \
#    --exclude 'work_dir' \
#    --exclude 'out_files' \
#    --exclude 'results_csiro' \
#    --exclude 'data' \
#    --exclude 'pretrained-models' \
#    --exclude 'Exp_Storage' \
#    --exclude 'Exp_Uncompleted' \
#    --exclude 'pretrain_eval' \
#    --exclude 'Motion-Data' \
#    --exclude 'results_dw_beijing' \
#    --exclude 'eval' \
#    --exclude 'ensemble_collection' \
#    --exclude 'visuals/' \
#    --exclude 'test_fields' \
#    --exclude 'IJCAI-Results'