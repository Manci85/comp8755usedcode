import argparse
import json
import pickle
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from metadata.class_labels import ntu120_code_labels


if __name__ == "__main__":
    alpha = 1

    label_path_ntu120_xsub = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_label.pkl'
    label_path_ntu120_xset = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/val_label.pkl'

    with open(label_path_ntu120_xset, 'rb') as label:
        label = np.array(pickle.load(label))

    out_1 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ensemle_collection/ntu120_xsub/original_joint/epoch1_test_score.pkl'
    out_2 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ensemle_collection/ntu120_xsub/original_bone/epoch1_test_score.pkl'

    out_3 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ensemle_collection/ntu120_xsub/20-11-28-ntu120_xsub_only_sgcn_temp_trans/epoch1_test_score.pkl'
    out_4 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ensemle_collection/ntu120_xsub/20-11-29-ntu120_xsub_only_g3d_temp_trans_split_5/epoch1_test_score.pkl'

    out_5 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ensemle_collection/ntu120_xsub/20-11-29-ntu120_joint_xsub_only_sgcn_temp_trans_split_5/epoch1_test_score.pkl'
    out_6 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/ensemle_collection/ntu120_xsub/20-11-29-ntu120_joint_xsub_only_g3d_temp_trans_split_5/epoch1_test_score.pkl'

    out_7 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-02-ntu120_xsub_only_sgcn_temp_trans_end_tcn_sigmoid/2020-12-02T21-04-44/epoch40_test_score.pkl'

    out_8 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-04-ntu120_xsub_joint_velocity_only_sgcn_temp_trans_end_tcn_sigmoid/2020-12-04T19-21-38/epoch35_test_score.pkl'
    out_9 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-04-ntu120_xsub_bone_velocity_only_sgcn_temp_trans_end_tcn_sigmoid/2020-12-04T09-39-57/epoch50_test_score.pkl'

    out_10 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-04-ntu120_xsub_joint_velocity_only_sgcn/2020-12-04T22-59-16/epoch50_test_score.pkl'
    out_11 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/20-12-04-ntu120_xsub_bone_velocity_only_sgcn/2020-12-04T18-32-41/epoch35_test_score.pkl'

    out_5_5_5 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_5_5_5/2020-12-05T08-17-25/epoch50_test_score.pkl'
    out_10_10_5 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_csiro/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_10_10_5/2020-12-06T10-01-31/epoch45_test_score.pkl'
    out_15_10_5 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_15_10_5/2020-12-05T22-01-09/epoch55_test_score.pkl'
    out_15_15_5 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_csiro/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_15_15_5/2020-12-06T09-22-22/epoch45_test_score.pkl'
    out_20_10_5 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_20_10_5/2020-12-06T11-33-50/epoch40_test_score.pkl'
    out_20_15_5 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_csiro/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_20_15_5/2020-12-06T09-28-28/epoch60_test_score.pkl'
    out_20_15_15 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_csiro/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_20_15_15/2020-12-06T09-35-41/epoch40_test_score.pkl'
    out_60_30_15 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_60_30_15/2020-12-05T22-24-47/epoch60_test_score.pkl'
    out_100_50_25 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_100_50_25/2020-12-06T09-36-38/epoch55_test_score.pkl'
    out_150_50_25 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_150_50_25/2020-12-06T13-36-50/epoch60_test_score.pkl'

    out_vel_trans_10_10_5 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-07-ntu120_xsub_bone_velocity_only_sgcn_temp_trans_10_10_5/2020-12-07T23-09-05/epoch30_test_score.pkl'
    out_vel_trans_20_15_5 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_csiro/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_20_15_15/2020-12-06T09-35-41/epoch40_test_score.pkl'
    out_vel_trans_75_30_15 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/20-12-07-ntu120_xsub_bone_velocity_only_sgcn_temp_trans_75_30_15/2020-12-08T01-28-44/epoch30_test_score.pkl'

    # 最初的bone stream, 只用SGCN
    out_bone_only_sgcn_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-09-ntu120_xsub_only_sgcn/2020-12-09T23-20-44/epoch40_test_score.pkl'
    out_bone_only_sgcn_1 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-09-ntu120_xsub_only_sgcn/2020-12-09T23-17-51/epoch45_test_score.pkl'
    out_bone_only_sgcn_2 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-09-ntu120_xsub_only_sgcn/2020-12-09T10-50-47/epoch45_test_score.pkl'

    with open(out_1, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(out_2, 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(out_3, 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(out_4, 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(out_5, 'rb') as r5:
        r5 = list(pickle.load(r5).items())

    with open(out_6, 'rb') as r6:
        r6 = list(pickle.load(r6).items())

    with open(out_7, 'rb') as r7:
        r7 = list(pickle.load(r7).items())

    with open(out_8, 'rb') as r8:
        r8 = list(pickle.load(r8).items())

    with open(out_9, 'rb') as r9:
        r9 = list(pickle.load(r9).items())

    with open(out_10, 'rb') as r_10:
        r_10 = list(pickle.load(r_10).items())

    with open(out_11, 'rb') as r_11:
        r_11 = list(pickle.load(r_11).items())

    with open(out_5_5_5, 'rb') as r_5_5_5:
        r_5_5_5 = list(pickle.load(r_5_5_5).items())

    with open(out_10_10_5, 'rb') as r_10_10_5:
        r_10_10_5 = list(pickle.load(r_10_10_5).items())

    with open(out_15_10_5, 'rb') as r_15_10_5:
        r_15_10_5 = list(pickle.load(r_15_10_5).items())

    with open(out_15_15_5, 'rb') as r_15_15_5:
        r_15_15_5 = list(pickle.load(r_15_15_5).items())

    with open(out_20_10_5, 'rb') as r_20_10_5:
        r_20_10_5 = list(pickle.load(r_20_10_5).items())

    with open(out_20_15_5, 'rb') as r_20_15_5:
        r_20_15_5 = list(pickle.load(r_20_15_5).items())

    with open(out_20_15_15, 'rb') as r_20_15_15:
        r_20_15_15 = list(pickle.load(r_20_15_15).items())

    with open(out_60_30_15, 'rb') as r_60_30_15:
        r_60_30_15 = list(pickle.load(r_60_30_15).items())

    with open(out_100_50_25, 'rb') as r_100_50_25:
        r_100_50_25 = list(pickle.load(r_100_50_25).items())

    with open(out_150_50_25, 'rb') as r_150_50_25:
        r_150_50_25 = list(pickle.load(r_150_50_25).items())

    with open(out_vel_trans_10_10_5, 'rb') as r_vel_trans_10_10_5:
        r_vel_trans_10_10_5 = list(pickle.load(r_vel_trans_10_10_5).items())

    with open(out_vel_trans_20_15_5, 'rb') as r_vel_trans_20_15_5:
        r_vel_trans_20_15_5 = list(pickle.load(r_vel_trans_20_15_5).items())

    with open(out_vel_trans_75_30_15, 'rb') as r_vel_trans_75_30_15:
        r_vel_trans_75_30_15 = list(pickle.load(r_vel_trans_75_30_15).items())

    # 就是把基础SGCN给聚合起来
    with open(out_bone_only_sgcn_0, 'rb') as r_bone_only_sgcn_0:
        r_bone_only_sgcn_0 = list(pickle.load(r_bone_only_sgcn_0).items())

    with open(out_bone_only_sgcn_1, 'rb') as r_bone_only_sgcn_1:
        r_bone_only_sgcn_1 = list(pickle.load(r_bone_only_sgcn_1).items())

    with open(out_bone_only_sgcn_2, 'rb') as r_bone_only_sgcn_2:
        r_bone_only_sgcn_2 = list(pickle.load(r_bone_only_sgcn_2).items())

    # 加上one hot和angle
    out_joint_bone_angle_onehot_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-09-ntu120_xsub_joint_bone_angle_only_sgcn/2020-12-11T17-30-05/epoch55_test_score.pkl'
    with open(out_joint_bone_angle_onehot_0, 'rb') as r_joint_bone_angle_onehot_0:
        r_joint_bone_angle_onehot_0 = list(pickle.load(r_joint_bone_angle_onehot_0).items())

    out_joint_bone_angle_onehot_1 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-09-ntu120_xsub_joint_bone_angle_only_sgcn/2020-12-12T06-38-11/epoch60_test_score.pkl'
    with open(out_joint_bone_angle_onehot_1, 'rb') as r_joint_bone_angle_onehot_1:
        r_joint_bone_angle_onehot_1 = list(pickle.load(r_joint_bone_angle_onehot_1).items())

    out_joint_angle_onehot_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-12-ntu120_xsub_joint_angle_only_sgcn/2020-12-12T06-04-46/epoch60_test_score.pkl'
    with open(out_joint_angle_onehot_0, 'rb') as r_joint_angle_onehot_0:
        r_joint_angle_onehot_0 = list(pickle.load(r_joint_angle_onehot_0).items())

    out_joint_bone_angle_onehot_velocity_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-09-ntu120_xsub_joint_bone_angle_velocity_only_sgcn/2020-12-12T20-50-45/epoch60_test_score.pkl'
    with open(out_joint_bone_angle_onehot_velocity_0, 'rb') as r_joint_bone_angle_onehot_velocity_0:
        r_joint_bone_angle_onehot_velocity_0 = list(pickle.load(r_joint_bone_angle_onehot_velocity_0).items())

    out_bone_angle_onehot_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-09-ntu120_xsub_bone_angle_only_sgcn/2020-12-12T20-29-13/epoch45_test_score.pkl'
    with open(out_bone_angle_onehot_0, 'rb') as r_bone_angle_onehot_0:
        r_bone_angle_onehot_0 = list(pickle.load(r_bone_angle_onehot_0).items())

    out_angle_2hands_onehot_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-17-ntu120_xsub_joint_bone_angle_cangle_hands_2hands_only_sgcn/2020-12-20T18-29-25/epoch97_test_score.pkl'
    with open(out_angle_2hands_onehot_0, 'rb') as r_angle_2hands_onehot_0:
        r_angle_2hands_onehot_0 = list(pickle.load(r_angle_2hands_onehot_0).items())

    out_angle_2hands_onehot_velocity_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-20-ntu120_xsub_joint_bone_angle_cangle_hands_2hands_onehot_velocity_only_sgcn/2020-12-20T10-47-15/epoch51_test_score.pkl'
    with open(out_angle_2hands_onehot_velocity_0, 'rb') as r_angle_2hands_onehot_velocity_0:
        r_angle_2hands_onehot_velocity_0 = list(pickle.load(r_angle_2hands_onehot_velocity_0).items())

    out_angle_2hands_xset_onehot_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-22-ntu120_xset_joint_bone_angle_cangle_hands_2hands_only_sgcn/2020-12-22T22-20-27/epoch50_test_score.pkl'
    with open(out_angle_2hands_xset_onehot_0, 'rb') as r_angle_2hands_xset_onehot_0:
        r_angle_2hands_xset_onehot_0 = list(pickle.load(r_angle_2hands_xset_onehot_0).items())

    out_angle_2hands_xset_onehot_velocity_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-22-ntu120_xset_joint_bone_angle_cangle_hands_2hands_velocity_only_sgcn/2020-12-23T07-53-37/epoch43_test_score.pkl'
    with open(out_angle_2hands_xset_onehot_velocity_0, 'rb') as r_angle_2hands_xset_onehot_velocity_0:
        r_angle_2hands_xset_onehot_velocity_0 = list(pickle.load(r_angle_2hands_xset_onehot_velocity_0).items())

    # 聚合ntu120 xsub joint, bone, angle
    # NTU120 XSub joint, bone, angle
    ntu120_xsub_joint = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-13-ntu120_xsub_joint_onehot_only_sgcn/2020-12-13T14-42-42/epoch50_test_score.pkl'
    ntu120_xsub_bone = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/20-12-29-ntu120_xsub_bone_only_sgcn_onehot/2020-12-30T10-19-02/epoch40_test_score.pkl'
    ntu120_xsub_angle = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-29-ntu120_xsub_angle_arms_legs_onehot_only_sgcn/2020-12-29T10-31-51/epoch30_test_score.pkl'
    ntu120_xsub_joint_velocity = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-01-ntu120_xsub_joint_velocity_onehot_only_sgcn/2021-01-01T10-37-59/epoch41_test_score.pkl'
    ntu120_xsub_bone_velocity = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-01-ntu120_xsub_bone_velocity_onehot_only_sgcn/2021-01-01T13-23-42/epoch32_test_score.pkl'
    ntu120_xsub_angle_velocity = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-31-ntu120_xsub_angle_hands_2hands_velocity_onehot_only_sgcn_adj_weight/2020-12-31T12-02-01/epoch35_test_score.pkl'

    with open(ntu120_xsub_joint, 'rb') as r_ntu120_xsub_joint:
        r_ntu120_xsub_joint = list(pickle.load(r_ntu120_xsub_joint).items())

    with open(ntu120_xsub_bone, 'rb') as r_ntu120_xsub_bone:
        r_ntu120_xsub_bone = list(pickle.load(r_ntu120_xsub_bone).items())

    with open(ntu120_xsub_angle, 'rb') as r_ntu120_xsub_angle:
        r_ntu120_xsub_angle = list(pickle.load(r_ntu120_xsub_angle).items())

    with open(ntu120_xsub_joint_velocity, 'rb') as r_ntu120_xsub_joint_velocity:
        r_ntu120_xsub_joint_velocity = list(pickle.load(r_ntu120_xsub_joint_velocity).items())

    with open(ntu120_xsub_bone_velocity, 'rb') as r_ntu120_xsub_bone_velocity:
        r_ntu120_xsub_bone_velocity = list(pickle.load(r_ntu120_xsub_bone_velocity).items())

    with open(ntu120_xsub_angle_velocity, 'rb') as r_ntu120_xsub_angle_velocity:
        r_ntu120_xsub_angle_velocity = list(pickle.load(r_ntu120_xsub_angle_velocity).items())

    # NTU120 xset
    ntu120_xset_joint = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/20-12-30-ntu120_xset_joint_onehot_only_sgcn/2020-12-31T09-09-04/epoch42_test_score.pkl'
    ntu120_xset_bone = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-30-ntu120_xset_bone_onehot_only_sgcn/2020-12-30T21-58-38/epoch52_test_score.pkl'
    ntu120_xset_angle = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-31-ntu120_xset_angle_arms_legs_onehot_velocity_only_sgcn/2020-12-31T08-24-03/epoch54_test_score.pkl'
    ntu120_xset_joint_v = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/21-01-01-ntu120_xset_joint_velocity_onehot_only_sgcn/2021-01-01T18-23-45/epoch55_test_score.pkl'
    ntu120_xset_bone_v = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/20-12-30-ntu120_xset_bone_velocity_onehot_only_sgcn/2021-01-02T02-44-42/epoch53_test_score.pkl'
    ntu120_xset_angle_v = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-31-ntu120_xset_angle_arms_legs_onehot_velocity_only_sgcn/2020-12-31T08-24-03/epoch54_test_score.pkl'
    with open(ntu120_xset_joint, 'rb') as r_ntu120_xset_joint:
        r_ntu120_xset_joint = list(pickle.load(r_ntu120_xset_joint).items())
    with open(ntu120_xset_bone, 'rb') as r_ntu120_xset_bone:
        r_ntu120_xset_bone = list(pickle.load(r_ntu120_xset_bone).items())
    with open(ntu120_xset_angle, 'rb') as r_ntu120_xset_angle:
        r_ntu120_xset_angle = list(pickle.load(r_ntu120_xset_angle).items())
    with open(ntu120_xset_joint_v, 'rb') as r_ntu120_xset_joint_v:
        r_ntu120_xset_joint_v = list(pickle.load(r_ntu120_xset_joint_v).items())
    with open(ntu120_xset_bone_v, 'rb') as r_ntu120_xset_bone_v:
        r_ntu120_xset_bone_v = list(pickle.load(r_ntu120_xset_bone_v).items())
    with open(ntu120_xset_angle_v, 'rb') as r_ntu120_xset_angle_v:
        r_ntu120_xset_angle_v = list(pickle.load(r_ntu120_xset_angle_v).items())

    correct_dict = defaultdict(list)

    true_labels = []
    predicted_labels = []

    # 我想visualize的错误结果
    monitored_wrong_ones = []

    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        # _, r11 = r1[i]
        # _, r22 = r2[i]
        # _, r33 = r3[i]
        # _, r44 = r4[i]
        # _, r55 = r5[i]
        # _, r66 = r6[i]
        # _, r77 = r7[i]
        # _, r88 = r8[i]
        # _, r99 = r9[i]
        # _, r1010 = r_10[i]
        # _, r1111 = r_11[i]

        # _, r_5_5_5_ = r_5_5_5[i]
        # _, r_10_10_5_ = r_10_10_5[i]
        # _, r_15_10_5_ = r_15_10_5[i]
        # _, r_15_15_5_ = r_15_15_5[i]
        # _, r_20_10_5_ = r_20_10_5[i]
        # _, r_20_15_5_ = r_20_15_5[i]
        # _, r_20_15_15_ = r_20_15_15[i]
        # _, r_60_30_15_ = r_60_30_15[i]
        # _, r_100_50_25_ = r_100_50_25[i]
        # _, r_150_50_25_ = r_150_50_25[i]

        # _, r_vel_trans_10_10_5_ = r_vel_trans_10_10_5[i]
        # _, r_vel_trans_20_15_5_ = r_vel_trans_20_15_5[i]
        # _, r_vel_trans_75_30_15_ = r_vel_trans_75_30_15[i]

        # _, r_bone_only_sgcn_0_ = r_bone_only_sgcn_0[i]
        # _, r_bone_only_sgcn_1_ = r_bone_only_sgcn_1[i]
        # _, r_bone_only_sgcn_2_ = r_bone_only_sgcn_2[i]

        # _, r_joint_bone_angle_onehot_0_ = r_joint_bone_angle_onehot_0[i]
        # _, r_joint_bone_angle_onehot_1_ = r_joint_bone_angle_onehot_1[i]
        # _, r_joint_bone_angle_onehot_velocity_0_ = r_joint_bone_angle_onehot_velocity_0[i]

        # _, r_joint_angle_onehot_0_ = r_joint_angle_onehot_0[i]

        # _, r_bone_angle_onehot_0_ = r_bone_angle_onehot_0[i]

        # _, r_angle_2hands_onehot_0_ = r_angle_2hands_onehot_0[i]
        # _, r_angle_2hands_onehot_velocity_0_ = r_angle_2hands_onehot_velocity_0[i]

        _, out_angle_2hands_xset_onehot_0_ = r_angle_2hands_xset_onehot_0[i]
        _, r_angle_2hands_xset_onehot_velocity_0_ = r_angle_2hands_xset_onehot_velocity_0[i]

        # _, r_ntu120_xsub_joint_ = r_ntu120_xsub_joint[i]
        # _, r_ntu120_xsub_bone_ = r_ntu120_xsub_bone[i]
        # _, r_ntu120_xsub_angle_ = r_ntu120_xsub_angle[i]
        # _, r_ntu120_xsub_joint_velocity_ = r_ntu120_xsub_joint_velocity[i]
        # _, r_ntu120_xsub_bone_velocity_ = r_ntu120_xsub_bone_velocity[i]
        # _, r_ntu120_xsub_angle_velocity_ = r_ntu120_xsub_angle_velocity[i]

        _, r_ntu120_xset_joint_ = r_ntu120_xset_joint[i]
        _, r_ntu120_xset_bone_ = r_ntu120_xset_bone[i]
        _, r_ntu120_xset_angle_ = r_ntu120_xset_angle[i]
        _, r_ntu120_xset_joint_velocity_ = r_ntu120_xset_joint_v[i]
        _, r_ntu120_xset_bone_velocity_ = r_ntu120_xset_bone_v[i]
        _, r_ntu120_xset_angle_velocity_ = r_ntu120_xset_angle_v[i]

        # r = 0.9 * (r55 + r66 + r33 + r44) + 1.1 * (r77 + r11 + r22)
        # r = r33 + r55 + r77 + r88 + r99
        # r = r33 + r55 + r77 + +r88 + r99 + r1212 + r1313 + r1414 + r11 + r22
        # r = r1212 + r1414 + r77 + r1313
        # r = r_5_5_5_ + r_10_10_5_ + r_15_10_5_ + r_15_15_5_ + r_20_10_5_ + r_60_30_15_ + \
        #     r_100_50_25_ + r_150_50_25_ + r55
        # r = 0.8 * (r_vel_trans_10_10_5_ + r_vel_trans_20_15_5_ + r_vel_trans_75_30_15_) + \
        #     1.2 * (r_10_10_5_ + r_15_10_5_ + r_15_15_5_ + r_20_10_5_ + r_60_30_15_ + r_100_50_25_)

        # r = r_bone_only_sgcn_0_ + r_bone_only_sgcn_1_ + r_bone_only_sgcn_2_
        # r = r_bone_only_sgcn_0_ + r33 + r_10_10_5_ + r_15_10_5_ + r_15_15_5_ + r_20_10_5_ + \
        #     r_60_30_15_ + r_100_50_25_ + r1010 + r1111
        # r = r_angle_2hands_onehot_0_ + r_angle_2hands_onehot_velocity_0_

        # r = r_joint_bone_angle_onehot_0_ \
        #     + r_joint_angle_onehot_0_ + r_joint_bone_angle_onehot_velocity_0_ + r_bone_angle_onehot_0_

        # r = out_angle_2hands_xset_onehot_0_ + r_angle_2hands_xset_onehot_velocity_0_
        # r = r_ntu120_xsub_joint_ + r_ntu120_xsub_bone_ + r_ntu120_xsub_angle_ + \
        #     + r_ntu120_xsub_joint_velocity_ + r_ntu120_xsub_bone_velocity_ + r_ntu120_xsub_angle_velocity_ + \
        #     + r_angle_2hands_onehot_0_ + r_angle_2hands_onehot_velocity_0_
        r = r_ntu120_xset_joint_ + r_ntu120_xset_bone_ + r_ntu120_xset_angle_ + \
            r_ntu120_xset_joint_velocity_ + r_ntu120_xset_bone_velocity_ + r_ntu120_xset_angle_velocity_ + \
            out_angle_2hands_xset_onehot_0_ + r_angle_2hands_xset_onehot_velocity_0_

        correct_dict[l].append(int(np.argmax(r) == int(l)))

        rank_5 = r.argsort()[-5:]

        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))

        true_labels.append(int(l))
        predicted_labels.append(r)

        if r == 71 - 1 and int(l) == 72 - 1:
            monitored_wrong_ones.append(int(i))

        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    if False:
        print('monitored wrong ones: ', monitored_wrong_ones)

        correct_dict_ = correct_dict.copy()
        for a_key in correct_dict:
            correct_dict[a_key] = '{:.2f}'.format(sum(correct_dict[a_key]) / len(correct_dict[a_key]))

        label_acc = {}
        for a_key in correct_dict:
            label_acc[ntu120_code_labels[int(a_key)+1]] = float(correct_dict[a_key])

        label_acc = dict(sorted(label_acc.items(), key=lambda item: item[1]))
        label_acc_keys = list(label_acc.keys())
        # print('label acc keys: ', label_acc_keys)

        # print(label_acc)
        conf_mat = confusion_matrix(true_labels, predicted_labels)

        most_confused = {}
        for i in correct_dict.keys():
            confusion_0 = np.argsort(conf_mat[int(i)])[::-1][0]
            confusion_1 = np.argsort(conf_mat[int(i)])[::-1][1]

            most_confused[ntu120_code_labels[int(i)+1]] = [
                "{}  {}".format(ntu120_code_labels[confusion_0 + 1], conf_mat[int(i)][confusion_0]),
                "{}  {}".format(ntu120_code_labels[confusion_1 + 1], conf_mat[int(i)][confusion_1]),
                "{}".format(len(correct_dict_[i]))
            ]
        most_confused_ = {}
        for i in label_acc_keys:
            most_confused_[i] = most_confused[i]

        print('label acc: ', json.dumps(label_acc, indent=4))
        print('most confused: ', json.dumps(most_confused_, indent=4))
        with open('analysis/confusion_matrix.json', 'w') as f:
            json.dump(most_confused_, f, indent=4)

        with open('analysis/accuracy_per_class.json', 'w') as f:
            json.dump(label_acc, f, indent=4)
