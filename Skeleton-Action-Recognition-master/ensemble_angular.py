import argparse
import json
import pickle
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from metadata.class_labels import ntu120_code_labels
from test_fields.kinetics_analysis import get_kinetics_dict


def open_result_file(a_path):
    with open(a_path, 'rb') as r_path:
        r_path = list(pickle.load(r_path).items())
    return r_path


if __name__ == "__main__":
    # labels
    label_path_ntu120_xsub = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_label.pkl'
    label_path_ntu120_xset = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xset/val_label.pkl'
    label_path_ntu_xsub = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_label.pkl'
    label_path_ntu_xview = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xview/val_label.pkl'
    label_path_kinetics = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/kinetics/val_label.pkl'

    to_print_confusion = True

    data_type = 'ntu120_xsub'
    the_label = None
    if data_type == 'ntu120_xsub':
        the_label = label_path_ntu120_xsub
    elif data_type == 'ntu120_xset':
        the_label = label_path_ntu120_xset
    elif data_type == 'kinetics':
        the_label = label_path_kinetics
    elif data_type == 'ntu_xsub':
        the_label = label_path_ntu_xsub
    elif data_type == 'ntu_xview':
        the_label = label_path_ntu_xview
    else:
        raise NotImplementedError

    with open(the_label, 'rb') as label:
        label = np.array(pickle.load(label))

    # kinetics original
    kinetics_orig_joint = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/eval/kinetics/joint/2021-01-05T22-45-27/epoch1_test_score.pkl'
    kinetics_orig_bone = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/eval/kinetics/bone/2021-01-05T22-37-05/epoch1_test_score.pkl'
    kinetics_orig_joint = open_result_file(kinetics_orig_joint)
    kinetics_orig_bone = open_result_file(kinetics_orig_bone)

    kinetics_jnt_ang_21_01_12 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-12-kinetics_joint_angle_arms_legs_onehot_only_sgcn/2021-01-14T07-31-01/epoch48_test_score.pkl')
    kinetics_bon_ang_21_01_12 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-12-kinetics_bone_angle_arms_legs_onehot_only_sgcn/2021-01-14T07-31-06/epoch56_test_score.pkl')

    # Experimental result files
    # ntu120_xsub_joint_angle_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-03-ntu120_xsub_joint_angle_arms_legs_onehot_only_sgcn/2021-01-04T12-07-30/epoch52_test_score.pkl'
    ntu120_xsub_joint_angle_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-17-ntu120_xsub_joint_bone_angle_cangle_hands_2hands_only_sgcn/2020-12-20T18-29-25/epoch97_test_score.pkl'
    # ntu120_xsub_bone_angle_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-03-ntu120_xsub_bone_angle_arms_legs_onehot_only_sgcn/2021-01-04T19-35-54/epoch56_test_score.pkl'
    ntu120_xsub_bone_angle_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-24-ntu120_xsub_joint_bone_angle_cangle_hands_2hands_adj_weight_only_sgcn/2020-12-27T21-23-49/epoch46_test_score.pkl'

    ntu120_xsub_joint = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-13-ntu120_xsub_joint_onehot_only_sgcn/2020-12-13T14-42-42/epoch50_test_score.pkl'
    ntu120_xsub_bone = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/20-12-29-ntu120_xsub_bone_only_sgcn_onehot/2020-12-30T10-19-02/epoch40_test_score.pkl'
    ntu120_xsub_angle = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-29-ntu120_xsub_angle_arms_legs_onehot_only_sgcn/2020-12-29T10-31-51/epoch30_test_score.pkl'

    # ntu120, xsub, velocity
    ntu120_xsub_joint_velocity = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-01-ntu120_xsub_joint_velocity_onehot_only_sgcn/2021-01-01T10-37-59/epoch41_test_score.pkl'
    ntu120_xsub_bone_velocity = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-01-ntu120_xsub_bone_velocity_onehot_only_sgcn/2021-01-01T13-23-42/epoch32_test_score.pkl'
    ntu120_xsub_angle_velocity = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-31-ntu120_xsub_angle_hands_2hands_velocity_onehot_only_sgcn_adj_weight/2020-12-31T12-02-01/epoch35_test_score.pkl'
    ntu120_xsub_joint_velocity = open_result_file(ntu120_xsub_joint_velocity)
    ntu120_xsub_bone_velocity = open_result_file(ntu120_xsub_bone_velocity)
    ntu120_xsub_angle_velocity = open_result_file(ntu120_xsub_angle_velocity)

    kinetics_joint_bone_angle_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-24-kinetics_joint_bone_angle_cangle_arms_legs_only_sgcn/2020-12-24T21-12-21/epoch56_test_score.pkl'
    kinetics_joint_bone_angle_v_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-27-kinetics_joint_bone_angle_cangle_arms_legs_only_sgcn_velocity/2020-12-27T16-10-14/epoch47_test_score.pkl'

    ntu120_xsub_joint_angle = open_result_file(ntu120_xsub_joint_angle_0)
    ntu120_xsub_bone_angle = open_result_file(ntu120_xsub_bone_angle_0)

    ntu120_xsub_joint = open_result_file(ntu120_xsub_joint)
    ntu120_xsub_bone = open_result_file(ntu120_xsub_bone)
    ntu120_xsub_angle = open_result_file(ntu120_xsub_angle)

    kinetics_joint_bone_angle_0 = open_result_file(kinetics_joint_bone_angle_0)
    kinetics_joint_bone_angle_v_0 = open_result_file(kinetics_joint_bone_angle_v_0)

    # ntu120 xsub
    ntu120_xsub_jnt_bon_ang_2hands = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-17-ntu120_xsub_joint_bone_angle_cangle_hands_2hands_only_sgcn/2020-12-20T18-29-25/epoch97_test_score.pkl'
    ntu120_xsub_jnt_bon_ang_2hands = open_result_file(ntu120_xsub_jnt_bon_ang_2hands)
    ntu120_xsub_jnt_bon_ang_2hands_v = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-20-ntu120_xsub_joint_bone_angle_cangle_hands_2hands_onehot_velocity_only_sgcn/2020-12-20T10-47-15/epoch51_test_score.pkl'
    ntu120_xsub_jnt_bon_ang_2hands_v = open_result_file(ntu120_xsub_jnt_bon_ang_2hands_v)

    ntu120_xsub_jnt_ang_2hands_21_01_03 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-03-ntu120_xsub_joint_angle_arms_legs_onehot_only_sgcn/2021-01-04T12-07-30/epoch52_test_score.pkl')
    ntu120_xsub_bon_ang_2hands_21_01_03 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-03-ntu120_xsub_bone_angle_arms_legs_onehot_only_sgcn/2021-01-04T19-35-54/epoch56_test_score.pkl')
    ntu120_xsub_jnt_ang_2hands_v_21_01_11 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-11-ntu120_xsub_joint_angle_v_hands_2hands_onehot_only_sgcn/2021-01-11T09-57-33/epoch56_test_score.pkl')
    ntu120_xsub_bon_ang_2hands_v_21_01_11 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-11-ntu120_xsub_bone_angle_v_hands_2hands_onehot_only_sgcn/2021-01-11T21-53-43/epoch41_test_score.pkl')

    ntu120_xsub_jnt_bon_ang_msg3d_v = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-20-ntu120_xsub_joint_bone_angle_v_hands_2hands_onehot_g3d/2021-01-20T00-24-18/epoch59_test_score.pkl')
    ntu120_xsub_jnt_bon_ang_msg3d = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-20-ntu120_xsub_joint_bone_angle_arms_legs_onehot_g3d/2021-01-20T00-09-40/epoch58_test_score.pkl')

    # ntu120 xset
    ntu120_xset_jnt_bon_ang_2hands = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-22-ntu120_xset_joint_bone_angle_cangle_hands_2hands_only_sgcn/2020-12-22T22-20-27/epoch50_test_score.pkl'
    ntu120_xset_jnt_bon_ang_2hands = open_result_file(ntu120_xset_jnt_bon_ang_2hands)
    ntu120_xset_jnt_bon_ang_2hands_v = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-22-ntu120_xset_joint_bone_angle_cangle_hands_2hands_velocity_only_sgcn/2020-12-23T07-53-37/epoch43_test_score.pkl'
    ntu120_xset_jnt_bon_ang_2hands_v = open_result_file(ntu120_xset_jnt_bon_ang_2hands_v)

    ntu120_xset_jnt_ang_2hands = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-11-ntu120_xset_joint_angle_arms_legs_velocity_onehot_only_sgcn/2021-01-11T21-54-53/epoch38_test_score.pkl')
    ntu120_xset_jnt_ang_2hands_v = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-11-ntu120_xset_joint_angle_arms_legs_velocity_onehot_only_sgcn/2021-01-11T21-54-53/epoch38_test_score.pkl')
    ntu120_xset_bon_ang_2hands = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-04-ntu120_xset_joint_angle_arms_legs_onehot_only_sgcn/2021-01-04T20-00-39/epoch53_test_score.pkl')
    ntu120_xset_bon_ang_2hands_v = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-04-ntu120_xset_bone_angle_arms_legs_onehot_only_sgcn/2021-01-04T22-27-57/epoch53_test_score.pkl')

    ntu120_xset_jnt_bon_ang_msg3d = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-22-ntu120_xset_jnt_bon_ang_arms_legs_1ht_g3d/2021-03-22T12-06-19/epoch58_test_score.pkl')
    ntu120_xset_jnt_bon_ang_msg3d_v = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-22-ntu120_xset_jnt_bon_ang_arms_legs_1ht_g3d/2021-03-22T19-13-14/epoch48_test_score.pkl')

    # kinetics results
    kin_jnt_bon_ang_arms_legs_head_21_01_06 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-06-kinetics_joint_bone_angle_cangle_arms_legs_head_only_sgcn/2021-01-06T21-51-04/epoch45_test_score.pkl'
    kin_jnt_bon_ang_arms_legs_head_21_01_06 = open_result_file(kin_jnt_bon_ang_arms_legs_head_21_01_06)

    # ntu, xsub
    ntu_xsub_jnt_bon_ang_arms_legs_21_01_05 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-05-ntu_xsub_joint_bone_angle_arms_legs_onehot_only_sgcn/2021-01-05T14-21-30/epoch55_test_score.pkl'
    ntu_xsub_jnt_bon_ang_arms_legs_21_01_05 = open_result_file(ntu_xsub_jnt_bon_ang_arms_legs_21_01_05)
    ntu_xsub_jnt_bon_ang_arms_legs_v_21_01_06 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-06-ntu_xsub_joint_bone_angle_arms_legs_velocity_onehot_only_sgcn/2021-01-06T08-20-45/epoch54_test_score.pkl'
    ntu_xsub_jnt_bon_ang_arms_legs_v_21_01_06 = open_result_file(ntu_xsub_jnt_bon_ang_arms_legs_v_21_01_06)
    ntu_xsub_jnt_ang_arms_legs_21_01_05 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-05-ntu_xsub_joint_angle_arms_legs_onehot_only_sgcn/2021-01-05T08-59-47/epoch38_test_score.pkl'
    ntu_xsub_jnt_ang_arms_legs_21_01_05 = open_result_file(ntu_xsub_jnt_ang_arms_legs_21_01_05)
    ntu_xsub_jnt_bon_ang_arms_legs_21_01_09 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-09-ntu_xsub_joint_bone_angle_arms_legs_onehot_only_sgcn/2021-01-09T13-29-15/epoch52_test_score.pkl')

    ntu_xsub_jnt_ang_hands_2hands_21_01_11 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-10-ntu_xsub_joint_angle_arms_legs_onehot_only_sgcn/2021-01-11T09-33-25/epoch38_test_score.pkl')
    ntu_xsub_jnt_ang_hands_2hands_v_21_01_11 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-10-ntu_xsub_joint_angle_arms_legs_velocity_onehot_only_sgcn/2021-01-10T20-58-17/epoch50_test_score.pkl')
    ntu_xsub_bon_ang_hands_2hands_21_01_11 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-10-ntu_xsub_bone_angle_arms_legs_onehot_only_sgcn/2021-01-11T09-33-29/epoch42_test_score.pkl')
    ntu_xsub_bon_ang_hands_2hands_v_21_01_11 = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-10-ntu_xsub_bone_angle_arms_legs_velocity_onehot_only_sgcn/2021-01-10T20-58-06/epoch44_test_score.pkl')

    # ntu, xsub, ft
    ntu_xsub_jnt_bon_ang_arms_legs_ft_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-08-ntu_xsub_joint_bone_angle_arms_legs_onehot_only_sgcn_ft/2021-01-08T21-07-03/epoch25_test_score.pkl'
    ntu_xsub_jnt_bon_ang_arms_legs_ft_0 = open_result_file(ntu_xsub_jnt_bon_ang_arms_legs_ft_0)

    # ntu, xsub, velocity, ft
    ntu_xsub_jnt_bon_ang_arms_legs_v_ft_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-08-ntu_xsub_joint_bone_angle_arms_legs_velocity_onehot_only_sgcn_ft/2021-01-08T12-19-07/epoch11_test_score.pkl'
    ntu_xsub_jnt_bon_ang_arms_legs_v_ft_0 = open_result_file(ntu_xsub_jnt_bon_ang_arms_legs_v_ft_0)

    # ntu, xsub, g3d
    ntu_xsub_jnt_bon_ang_arms_legs_g3d = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-20-ntu_xsub_jnt_bon_ang_arms_legs_1ht_g3d/2021-03-20T22-09-04/epoch51_test_score.pkl')
    ntu_xsub_jnt_bon_ang_arms_legs_v_g3d = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-20-ntu_xsub_jnt_bon_ang_arms_legs_1ht_g3d/2021-03-20T22-09-18/epoch60_test_score.pkl')

    # ntu, xview
    ntu_xview_jnt_bon_ang_arms_legs_21_01_05 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-05-ntu_xview_joint_bone_angle_arms_legs_onehot_only_sgcn/2021-01-05T23-51-00/epoch49_test_score.pkl'
    ntu_xview_jnt_bon_ang_arms_legs_21_01_05 = open_result_file(ntu_xview_jnt_bon_ang_arms_legs_21_01_05)
    ntu_xview_jnt_bon_ang_arms_legs_v_21_01_06 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-06-ntu_xview_joint_bone_angle_arms_legs_velocity_onehot_only_sgcn/2021-01-06T17-15-06/epoch59_test_score.pkl'
    ntu_xview_jnt_bon_ang_arms_legs_v_21_01_06 = open_result_file(ntu_xview_jnt_bon_ang_arms_legs_v_21_01_06)
    ntu_xview_jnt_bon_ang_arms_legs_21_01_09 = open_result_file(
        'results_dw_beijing/21-01-09-ntu_xview_joint_bone_angle_arms_legs_onehot_only_sgcn/'
        '2021-01-09T13-29-51/epoch46_test_score.pkl'
    )
    ntu_xview_jnt_bon_ang_arms_legs_v_21_01_09 = open_result_file(
        'results_dw_beijing/21-01-09-ntu_xview_joint_bone_angle_arms_legs_velocity_onehot_only_sgcn/'
        '2021-01-09T13-30-48/epoch63_test_score.pkl'
    )

    ntu_xview_jnt_ang_arms_legs_21_01_10 = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/'
        '21-01-10-ntu_xview_joint_angle_arms_legs_onehot_only_sgcn/2021-01-10T09-25-56/epoch55_test_score.pkl'
    )

    ntu_xview_jnt_ang_v_arms_legs_21_01_10 = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/'
        '21-01-10-ntu_xview_joint_angle_arms_legs_velocity_onehot_only_sgcn/2021-01-10T09-25-23/epoch44_test_score.pkl'
    )

    ntu_xview_bon_ang_arms_legs_21_01_10 = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/'
        '21-01-10-ntu_xview_bone_angle_arms_legs_onehot_only_sgcn/2021-01-10T20-54-46/epoch45_test_score.pkl'
    )

    ntu_xview_bon_ang_v_arms_legs_21_01_10 = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/'
        '21-01-10-ntu_xview_bone_angle_arms_legs_velocity_onehot_only_sgcn/2021-01-10T09-25-12/epoch49_test_score.pkl'
    )

    # ntu, xview, ft
    ntu_xview_jnt_bon_ang_arms_legs_ft_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-08-ntu_xview_joint_bone_angle_arms_legs_onehot_only_sgcn_ft/2021-01-08T21-46-31/epoch25_test_score.pkl'
    ntu_xview_jnt_bon_ang_arms_legs_ft_0 = open_result_file(ntu_xview_jnt_bon_ang_arms_legs_ft_0)

    # ntu, xview, ft
    ntu_xview_jnt_bon_ang_arms_legs_ft_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-08-ntu_xview_joint_bone_angle_arms_legs_onehot_only_sgcn_ft/2021-01-08T21-46-31/epoch25_test_score.pkl'
    ntu_xview_jnt_bon_ang_arms_legs_ft_0 = open_result_file(ntu_xview_jnt_bon_ang_arms_legs_ft_0)

    # ntu, xview, velocity, ft
    ntu_xview_jnt_bon_ang_arms_legs_v_ft_0 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-08-ntu_xview_joint_bone_angle_arms_legs_velocity_onehot_only_sgcn_ft/2021-01-08T21-09-40/epoch4_test_score.pkl'
    ntu_xview_jnt_bon_ang_arms_legs_v_ft_0 = open_result_file(ntu_xview_jnt_bon_ang_arms_legs_v_ft_0)

    # Table filling
    ## NTU XSub
    ntu_xsub_jnt_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/21-03-14-ntu_xsub_joint_only_sgcn/test/2021-03-14T00-25-11/epoch29_test_score.pkl'
    ntu_xsub_jnt_orig = open_result_file(ntu_xsub_jnt_orig)

    ntu_xsub_bon_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/21-03-14-ntu_xsub_bone_only_sgcn/2021-03-14T09-25-25/epoch29_test_score.pkl'
    ntu_xsub_bon_orig = open_result_file(ntu_xsub_bon_orig)

    ntu_xsub_jnt_ang_arms_legs_v = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-25-ntu_xsub_jnt_vel_1ht_only_sgcn/2021-05-25T19-18-11/epoch40_test_score.pkl')
    ntu_xsub_bon_ang_arms_legs_v = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-25-ntu_xsub_bon_vel_1ht_only_sgcn/2021-05-25T19-37-05/epoch40_test_score.pkl')

    ntu_xsub_jnt_bon_arms_legs = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-26-ntu_xsub_jnt_bon_1ht_only_sgcn/2021-05-26T15-03-10/epoch40_test_score.pkl')
    ntu_xsub_jnt_bon_arms_legs_v = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-26-ntu_xsub_jnt_bon_vel_1ht_only_sgcn/2021-05-26T16-39-10/epoch40_test_score.pkl')

    ## NTU XView
    ntu_xview_jnt_bon_v = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-27-ntu_xview_jnt_bon_vel_1ht_only_sgcn/2021-05-27T00-15-26/epoch55_test_score.pkl')
    ntu_xview_jnt_bon = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-27-ntu_xview_jnt_bon_1ht_only_sgcn/2021-05-27T09-42-57/epoch50_test_score.pkl')
    ntu_xview_bon_v = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-27-ntu_xview_bon_vel_1ht_only_sgcn/2021-05-27T17-43-09/epoch35_test_score.pkl')
    ntu_xview_jnt_v = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-27-ntu_xview_jnt_vel_1ht_only_sgcn/2021-05-28T07-52-59/epoch50_test_score.pkl')

    ## NTU120 XSet
    ntu120_xset_jnt_bon_vel = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-27-ntu120_xset_jnt_bon_vel_1ht_only_sgcn/2021-05-27T00-36-09/epoch40_test_score.pkl')
    ntu120_xset_jnt_bon = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-27-ntu120_xset_jnt_bon_1ht_only_sgcn/2021-05-28T20-01-54/epoch55_test_score.pkl')
    ntu120_xset_bon_vel = open_result_file('/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-27-ntu120_xset_bon_vel_1ht_only_sgcn/2021-05-28T07-57-59/epoch30_test_score.pkl')

    # Obtaining pretrained models
    ## NTU 60 XSub
    # ntu60_xsub_jnt_bon_ang_21_05_30 = open_result_file()

    # Error analysis
    correct_dict = defaultdict(list)
    true_labels = []
    predicted_labels = []

    # 我想visualize的错误结果
    monitored_wrong_ones = []

    # correct analysis
    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        if data_type == 'ntu120_xsub':
            _, ntu120_xsub_joint_angle_ = ntu120_xsub_joint_angle[i]
            _, ntu120_xsub_bone_angle_ = ntu120_xsub_bone_angle[i]
            _, ntu120_xsub_joint_ = ntu120_xsub_joint[i]
            _, ntu120_xsub_bone_ = ntu120_xsub_bone[i]
            _, ntu120_xsub_angle_ = ntu120_xsub_angle[i]

            _, ntu120_xsub_joint_velocity_ = ntu120_xsub_joint_velocity[i]
            _, ntu120_xsub_bone_velocity_ = ntu120_xsub_bone_velocity[i]
            _, ntu120_xsub_angle_velocity_ = ntu120_xsub_angle_velocity[i]

            _, ntu120_xsub_jnt_bon_ang_2hands_ = ntu120_xsub_jnt_bon_ang_2hands[i]
            _, ntu120_xsub_jnt_bon_ang_2hands_v_ = ntu120_xsub_jnt_bon_ang_2hands_v[i]

            _, ntu120_xsub_jnt_ang_2hands_21_01_03_ = ntu120_xsub_jnt_ang_2hands_21_01_03[i]
            _, ntu120_xsub_bon_ang_2hands_21_01_03_ = ntu120_xsub_bon_ang_2hands_21_01_03[i]
            _, ntu120_xsub_jnt_ang_2hands_v_21_01_11_ = ntu120_xsub_jnt_ang_2hands_v_21_01_11[i]
            _, ntu120_xsub_bon_ang_2hands_v_21_01_11_ = ntu120_xsub_bon_ang_2hands_v_21_01_11[i]

            _, ntu120_xsub_jnt_bon_ang_msg3d_ = ntu120_xsub_jnt_bon_ang_msg3d[i]
            _, ntu120_xsub_jnt_bon_ang_msg3d_v_ = ntu120_xsub_jnt_bon_ang_msg3d_v[i]
            # r = ntu120_xsub_joint_angle_ + ntu120_xsub_bone_angle_ + \
            #     ntu120_xsub_jnt_bon_ang_2hands_ + ntu120_xsub_jnt_bon_ang_2hands_v_
            # r = ntu120_xsub_jnt_bon_ang_msg3d_ + ntu120_xsub_jnt_bon_ang_msg3d_v_ + \
            #     ntu120_xsub_joint_angle_ + ntu120_xsub_bone_angle_
            # r = ntu120_xsub_jnt_bon_ang_msg3d_v_
            # r = ntu120_xsub_bone_ + ntu120_xsub_bone_velocity_
            # r = ntu120_xsub_bone_ + ntu120_xsub_bone_velocity_
            r = ntu120_xsub_jnt_ang_2hands_v_21_01_11_

        elif data_type == 'ntu120_xset':
            _, ntu120_xset_jnt_bon_ang_2hands_ = ntu120_xset_jnt_bon_ang_2hands[i]
            _, ntu120_xset_jnt_bon_ang_2hands_v_ = ntu120_xset_jnt_bon_ang_2hands_v[i]

            _, ntu120_xset_jnt_ang_2hands_ = ntu120_xset_jnt_ang_2hands[i]
            _, ntu120_xset_jnt_ang_2hands_v_ = ntu120_xset_jnt_ang_2hands_v[i]
            _, ntu120_xset_bon_ang_2hands_ = ntu120_xset_bon_ang_2hands[i]
            _, ntu120_xset_bon_ang_2hands_v_ = ntu120_xset_bon_ang_2hands_v[i]

            _, ntu120_xset_jnt_bon_ang_msg3d_ = ntu120_xset_jnt_bon_ang_msg3d[i]
            _, ntu120_xset_jnt_bon_ang_vel_msg3d_ = ntu120_xset_jnt_bon_ang_msg3d_v[i]

            _, ntu120_xset_bon_vel_ = ntu120_xset_bon_vel[i]

            # r = ntu120_xset_jnt_bon_ang_2hands_v_ + ntu120_xset_jnt_bon_ang_2hands_
            # r = ntu120_xset_jnt_ang_2hands_ + \
            #     ntu120_xset_bon_ang_2hands_ + \
            #     ntu120_xset_jnt_bon_ang_2hands_v_ + ntu120_xset_jnt_bon_ang_2hands_
            # r = ntu120_xset_jnt_bon_ang_2hands_ + ntu120_xset_jnt_bon_ang_2hands_v_ + \
            #     ntu120_xset_bon_ang_2hands_
            # r = ntu120_xset_jnt_bon_ang_msg3d_ + ntu120_xset_jnt_bon_ang_vel_msg3d_
            r = ntu120_xset_bon_ang_2hands_ + ntu120_xset_bon_ang_2hands_v_

        elif data_type == 'kinetics':
            _, kinetics_joint_bone_angle_0_ = kinetics_joint_bone_angle_0[i]
            _, kinetics_joint_bone_angle_0_v_ = kinetics_joint_bone_angle_v_0[i]
            _, kinetics_orig_bone_ = kinetics_orig_joint[i]
            _, kin_jnt_bon_ang_arms_legs_head_21_01_06_ = kin_jnt_bon_ang_arms_legs_head_21_01_06[i]
            _, kinetics_jnt_ang_21_01_12_ = kinetics_orig_joint[i]
            _, kinetics_bon_ang_21_01_12_ = kinetics_orig_bone[i]

            r = kinetics_joint_bone_angle_0_ + kinetics_joint_bone_angle_0_v_

        elif data_type == 'ntu_xsub':
            _, ntu_jnt_bon_ang_arms_legs_21_01_05_ = ntu_xsub_jnt_bon_ang_arms_legs_21_01_05[i]
            _, ntu_jnt_bon_ang_arms_legs_v_21_01_05_ = ntu_xsub_jnt_bon_ang_arms_legs_v_21_01_06[i]
            _, ntu_xsub_jnt_ang_arms_legs_21_01_05_ = ntu_xsub_jnt_ang_arms_legs_21_01_05[i]

            _, ntu_xsub_jnt_bon_ang_arms_legs_ft_0_ = ntu_xsub_jnt_bon_ang_arms_legs_ft_0[i]
            _, ntu_xsub_jnt_bon_ang_arms_legs_v_ft_0_ = ntu_xsub_jnt_bon_ang_arms_legs_v_ft_0[i]
            _, ntu_xsub_jnt_bon_ang_arms_legs_21_01_09_ = ntu_xsub_jnt_bon_ang_arms_legs_21_01_09[i]

            _, ntu_xsub_jnt_ang_hands_2hands_21_01_11_ = ntu_xsub_jnt_ang_hands_2hands_21_01_11[i]
            _, ntu_xsub_bon_ang_hands_2hands_21_01_11_ = ntu_xsub_bon_ang_hands_2hands_21_01_11[i]
            _, ntu_xsub_jnt_ang_hands_2hands_v_21_01_11_ = ntu_xsub_jnt_ang_hands_2hands_v_21_01_11[i]
            _, ntu_xsub_bon_ang_hands_2hands_v_21_01_11_ = ntu_xsub_bon_ang_hands_2hands_v_21_01_11[i]

            _, ntu_xsub_jnt_bon_ang_arms_legs_g3d_ = ntu_xsub_jnt_bon_ang_arms_legs_g3d[i]
            _, ntu_xsub_jnt_bon_ang_arms_legs_v_g3d_ = ntu_xsub_jnt_bon_ang_arms_legs_v_g3d[i]

            _, ntu_xsub_jnt_bon_arms_legs_ = ntu_xsub_jnt_bon_arms_legs[i]
            _, ntu_xsub_jnt_bon_arms_legs_v_ = ntu_xsub_jnt_bon_arms_legs_v[i]
            _, ntu_xsub_bon_orig_ = ntu_xsub_bon_orig[i]
            _, ntu_xsub_jnt_orig_ = ntu_xsub_jnt_orig[i]

            # r = ntu_xsub_jnt_bon_ang_arms_legs_as21_01_09_ + ntu_xsub_jnt_bon_ang_arms_legs_ft_0_ + ntu_xsub_jnt_bon_ang_arms_legs_v_ft_0_ + ntu_xsub_jnt_ang_arms_legs_21_01_05_
            # r = ntu_xsub_jnt_ang_arms_legs_21_01_05_ + ntu_xsub_bon_ang_hands_2hands_21_01_11_ + \
            #     ntu_xsub_jnt_ang_hands_2hands_v_21_01_11_ + ntu_xsub_bon_ang_hands_2hands_v_21_01_11_
            # r = ntu_xsub_jnt_ang_arms_legs_21_01_05_ + ntu_xsub_bon_ang_hands_2hands_21_01_11_ + \
            #     (ntu_xsub_jnt_ang_hands_2hands_v_21_01_11_ + ntu_xsub_bon_ang_hands_2hands_v_21_01_11_) + \
            #     + ntu_jnt_bon_ang_arms_legs_21_01_05_ + ntu_jnt_bon_ang_arms_legs_v_21_01_05_
            # r = ntu_xsub_jnt_ang_hands_2hands_21_01_11_ + ntu_xsub_bon_ang_hands_2hands_21_01_11_ + \
            #     ntu_jnt_bon_ang_arms_legs_21_01_05_ + ntu_jnt_bon_ang_arms_legs_v_21_01_05_
            # r = ntu_xsub_jnt_bon_ang_arms_legs_g3d_ + ntu_xsub_jnt_bon_ang_arms_legs_v_g3d_

            # r = ntu_xsub_jnt_orig_ + ntu_xsub_bon_orig_[:60] + ntu_xsub_jnt_bon_arms_legs_[:60] + ntu_xsub_jnt_bon_arms_legs_v_[:60]
            r = ntu_jnt_bon_ang_arms_legs_21_01_05_

        elif data_type == 'ntu_xview':
            _, ntu_xview_jnt_bon_ang_arms_legs_21_01_05_ = ntu_xview_jnt_bon_ang_arms_legs_21_01_05[i]
            _, ntu_xview_jnt_bon_ang_arms_legs_v_21_01_05_ = ntu_xview_jnt_bon_ang_arms_legs_v_21_01_06[i]

            _, ntu_xview_jnt_bon_ang_arms_legs_21_01_09_ = ntu_xview_jnt_bon_ang_arms_legs_21_01_09[i]
            _, ntu_xview_jnt_bon_ang_arms_legs_v_21_01_09_ = ntu_xview_jnt_bon_ang_arms_legs_v_21_01_09[i]

            _, ntu_xview_jnt_bon_ang_arms_legs_ft_0_ = ntu_xview_jnt_bon_ang_arms_legs_ft_0[i]
            _, ntu_xview_jnt_bon_ang_arms_legs_v_ft_0_ = ntu_xview_jnt_bon_ang_arms_legs_v_ft_0[i]

            _, ntu_xview_jnt_ang_arms_legs_21_01_10_ = ntu_xview_jnt_ang_arms_legs_21_01_10[i]
            _, ntu_xview_bon_ang_arms_legs_21_01_10_ = ntu_xview_bon_ang_arms_legs_21_01_10[i]
            _, ntu_xview_jnt_ang_v_arms_legs_21_01_10_ = ntu_xview_jnt_ang_v_arms_legs_21_01_10[i]
            _, ntu_xview_bon_ang_v_arms_legs_21_01_10_ = ntu_xview_bon_ang_v_arms_legs_21_01_10[i]

            _, ntu_xview_jnt_bon_ = ntu_xview_jnt_bon[i]
            _, ntu_xview_jnt_bon_v_ = ntu_xview_jnt_bon_v[i]
            _, ntu_xview_bon_v_ = ntu_xview_bon_v[i]
            _, ntu_xview_jnt_v_ = ntu_xview_jnt_v[i]
            # _, ntu_xsub_jnt_ang_arms_legs_v_ = ntu_xsub_jnt_ang_arms_legs_v[i]
            # _, ntu_xsub_bon_ang_arms_legs_v_ = ntu_xsub_bon_ang_arms_legs_v[i]

            # r = ntu_xview_jnt_bon_ang_arms_legs_21_01_05_ + ntu_xview_jnt_bon_ang_arms_legs_v_21_01_05_
            # r = ntu_xview_jnt_ang_arms_legs_21_01_10_ + ntu_xview_bon_ang_arms_legs_21_01_10_ + \
            #     ntu_xview_jnt_bon_ang_arms_legs_21_01_05_ + ntu_xview_jnt_bon_ang_arms_legs_v_21_01_05_
            r = ntu_xview_jnt_bon_ + ntu_xview_jnt_bon_v_ + ntu_xview_bon_v_ + ntu_xview_jnt_v_

        correct_dict[l].append(int(np.argmax(r) == int(l)))

        rank_5 = r.argsort()[-5:]

        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))

        true_labels.append(int(l))
        predicted_labels.append(r)

        total_num += 1

    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    if to_print_confusion:
        print('monitored wrong ones: ', monitored_wrong_ones)
        if 'ntu' in data_type:
            code_labels = ntu120_code_labels
        elif 'kinetics' in data_type:
            code_labels = get_kinetics_dict()
        else:
            raise NotImplementedError

        correct_dict_ = correct_dict.copy()
        for a_key in correct_dict:
            correct_dict[a_key] = '{:.6f}'.format(sum(correct_dict[a_key]) / len(correct_dict[a_key]))

        label_acc = {}
        for a_key in correct_dict:
            label_acc[code_labels[int(a_key)+1]] = float(correct_dict[a_key])

        label_acc = dict(sorted(label_acc.items(), key=lambda item: item[1]))
        label_acc_keys = list(label_acc.keys())
        # print('label acc keys: ', label_acc_keys)

        # print(label_acc)
        conf_mat = confusion_matrix(true_labels, predicted_labels)

        most_confused = {}
        for i in correct_dict.keys():
            confusion_0 = np.argsort(conf_mat[int(i)])[::-1][0]
            confusion_1 = np.argsort(conf_mat[int(i)])[::-1][1]

            most_confused[code_labels[int(i)+1]] = [
                "{}  {}".format(code_labels[confusion_0 + 1], conf_mat[int(i)][confusion_0]),
                "{}  {}".format(code_labels[confusion_1 + 1], conf_mat[int(i)][confusion_1]),
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