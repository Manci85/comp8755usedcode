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

import torch

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

    to_print_confusion = False

    data_type = 'ntu_xview'
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

    # ntu120 xsub
    ## Baselines
    ntu120_xsub_jnt_1ht = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-27-ntu_xsub_jnt_1ht_hyp/2021-01-28T09-44-17/epoch34_test_score.pkl'
    ntu120_xsub_jnt_1ht = open_result_file(ntu120_xsub_jnt_1ht)

    ntu120_xsub_bon_1ht = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/20-12-05-ntu120_xsub_bone_only_sgcn_temp_trans_end_tcn_sigmoid_section_20_15_15/2020-12-06T09-36-38/epoch35_test_score.pkl'
    ntu120_xsub_bon_1ht = open_result_file(ntu120_xsub_bon_1ht)

    ## DCT
    ntu120_xsub_jnt_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-04-ntu120_xsub_jnt_nerf_2_1ht_only_sgcn/2021-03-04T22-30-42/epoch60_test_score.pkl'
    ntu120_xsub_jnt_dct = open_result_file(ntu120_xsub_jnt_dct)

    ntu120_xsub_bon_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-04-ntu120_xsub_bon_nerf_2_1ht_only_sgcn/2021-03-04T22-33-57/epoch53_test_score.pkl'
    ntu120_xsub_bon_dct = open_result_file(ntu120_xsub_bon_dct)

    ntu120_xsub_jnt_vel_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-09-ntu120_xsub_jnt_nerf_vel_2_1ht_only_sgcn/2021-03-09T10-12-29/epoch43_test_score.pkl'
    ntu120_xsub_jnt_vel_dct = open_result_file(ntu120_xsub_jnt_vel_dct)

    ntu120_xsub_bon_vel_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-09-ntu120_xsub_bon_nerf_vel_2_1ht_only_sgcn/2021-03-09T10-25-28/epoch46_test_score.pkl'
    ntu120_xsub_bon_vel_dct = open_result_file(ntu120_xsub_bon_vel_dct)

    ## Nerf
    ntu120_xsub_jnt_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-02-24-ntu120_xsub_jnt_nerf_2_1ht_only_sgcn/2021-02-24T18-47-30/epoch50_test_score.pkl'
    ntu120_xsub_jnt_nerf = open_result_file(ntu120_xsub_jnt_nerf)

    ntu120_xsub_bon_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-02-24-ntu120_xsub_bon_nerf_2_1ht_only_sgcn/2021-02-24T18-52-36/epoch44_test_score.pkl'
    ntu120_xsub_bon_nerf = open_result_file(ntu120_xsub_bon_nerf)

    ## MSG3D
    ntu120_xsub_jnt_ste_msg3d = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-14-ntu120_xsub_2020_cvpr_msg3d_ste_k_3/2021-09-15T11-25-59/epoch46_test_score.pkl'
    ntu120_xsub_jnt_ste_msg3d = open_result_file(ntu120_xsub_jnt_ste_msg3d)

    ntu120_xsub_jnt_tte_msg3d = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-14-ntu120_xsub_2020_cvpr_msg3d_tte_k_3/2021-09-15T10-40-17/epoch56_test_score.pkl'
    ntu120_xsub_jnt_tte_msg3d = open_result_file(ntu120_xsub_jnt_tte_msg3d)

    ntu120_xsub_bon_ste_msg3d = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-16-ntu120_xsub_bone_2020_cvpr_msg3d_ste_k_3/2021-09-16T05-05-05/epoch50_test_score.pkl'
    ntu120_xsub_bon_ste_msg3d = open_result_file(ntu120_xsub_bon_ste_msg3d)

    ntu120_xsub_bon_tte_msg3d = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-16-ntu120_xsub_bone_2020_cvpr_msg3d_tte_k_3/2021-09-16T05-03-52/epoch50_test_score.pkl'
    ntu120_xsub_bon_tte_msg3d = open_result_file(ntu120_xsub_bon_tte_msg3d)

    # ntu120 xset
    ## DCT
    ntu120_xset_jnt_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-06-ntu120_xset_jnt_nerf_2_1ht_only_sgcn/2021-03-06T20-18-49/epoch56_test_score.pkl'
    ntu120_xset_jnt_dct = open_result_file(ntu120_xset_jnt_dct)

    ntu120_xset_bon_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-06-ntu120_xset_bon_nerf_2_1ht_only_sgcn/2021-03-06T20-19-00/epoch52_test_score.pkl'
    ntu120_xset_bon_dct = open_result_file(ntu120_xset_bon_dct)

    ## Nerf
    ntu120_xset_jnt_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-09-ntu120_xset_jnt_nerf_orig_2_1ht_only_sgcn/2021-03-09T21-37-29/epoch43_test_score.pkl'
    ntu120_xset_jnt_nerf = open_result_file(ntu120_xset_jnt_nerf)

    ntu120_xset_bon_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-09-ntu120_xset_bon_nerf_orig_2_1ht_only_sgcn/2021-03-09T21-37-41/epoch45_test_score.pkl'
    ntu120_xset_bon_nerf = open_result_file(ntu120_xset_bon_nerf)

    ## Baseline
    ntu120_xset_jnt_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-14-ntu120_xset_jnt_only_sgcn/2021-03-14T13-09-48/epoch30_test_score.pkl'
    ntu120_xset_jnt_orig = open_result_file(ntu120_xset_jnt_orig)

    ntu120_xset_bon_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-14-ntu120_xset_bon_only_sgcn/2021-03-14T13-10-21/epoch30_test_score.pkl'
    ntu120_xset_bon_orig = open_result_file(ntu120_xset_bon_orig)

    # kinetics results
    kin_jnt_bon_ang_arms_legs_head_21_01_06 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-06-kinetics_joint_bone_angle_cangle_arms_legs_head_only_sgcn/2021-01-06T21-51-04/epoch45_test_score.pkl'
    kin_jnt_bon_ang_arms_legs_head_21_01_06 = open_result_file(kin_jnt_bon_ang_arms_legs_head_21_01_06)

    #### MSG3D
    msg3d_ntu120_xset_joint_tte_k_3 = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-16-ntu120_xset_joint_2020_cvpr_msg3d_tte_k_3/2021-09-17T12-34-06/epoch60_test_score.pkl'
    )
    msg3d_ntu120_xset_bon_ste_k_3 = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-16-ntu120_xset_bone_2020_cvpr_msg3d_ste_k_3/2021-09-18T18-16-38/epoch45_test_score.pkl'
    )
    msg3d_ntu120_xset_bone_tte_k_3 = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-16-ntu120_xset_bone_2020_cvpr_msg3d_tte_k_3/2021-09-18T18-16-14/epoch55_test_score.pkl'
    )

    # ntu, xsub
    ## DCT
    # ntu_xsub_jnt_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-07-ntu_xsub_jnt_nerf_2_1ht_only_sgcn/2021-03-07T12-23-29/epoch57_test_score.pkl'
    ntu_xsub_jnt_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-15-ntu_xsub_jnt_nerf_2_1ht_only_sgcn/2021-03-15T07-21-52/epoch37_test_score.pkl'
    ntu_xsub_jnt_dct = open_result_file(ntu_xsub_jnt_dct)
    # ntu_xsub_bon_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-07-ntu_xsub_bon_nerf_2_1ht_only_sgcn/2021-03-07T12-23-32/epoch54_test_score.pkl'
    ntu_xsub_bon_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-15-ntu_xsub_bon_nerf_2_1ht_only_sgcn/2021-03-14T22-11-33/epoch37_test_score.pkl'
    # ntu_xsub_bon_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-15-ntu_xsub_bon_nerf_2_1ht_only_sgcn/2021-03-14T22-11-33/epoch37_test_score.pkl'
    ntu_xsub_bon_dct = open_result_file(ntu_xsub_bon_dct)

    ## Nerf
    ntu_xsub_jnt_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-10-ntu_xsub_jnt_nerf_orig_2_1ht_only_sgcn/2021-03-10T12-10-46/epoch33_test_score.pkl'
    ntu_xsub_jnt_nerf = open_result_file(ntu_xsub_jnt_nerf)
    # ntu_xsub_bon_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-10-ntu_xsub_bon_nerf_orig_2_1ht_only_sgcn/2021-03-10T12-07-41/epoch38_test_score.pkl'
    ntu_xsub_bon_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-16-ntu_xsub_bon_nerf_orig_4_1ht_only_sgcn/2021-03-16T20-32-54/epoch38_test_score.pkl'
    ntu_xsub_bon_nerf = open_result_file(ntu_xsub_bon_nerf)

    ## MSG3D
    ntu_xsub_jnt_dct_msg3d = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-11-ntu_xsub_jnt_nerf_dct_2_1ht_msg3d/2021-03-11T18-15-32/epoch40_test_score.pkl'
    ntu_xsub_jnt_dct_msg3d = open_result_file(ntu_xsub_jnt_dct_msg3d)

    ntu_xsub_bon_dct_msg3d = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-11-ntu_xsub_bon_nerf_dct_2_1ht_msg3d/2021-03-11T18-09-16/epoch55_test_score.pkl'
    ntu_xsub_bon_dct_msg3d = open_result_file(ntu_xsub_bon_dct_msg3d)

    ## Baseline
    ntu_xsub_jnt_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/21-03-14-ntu_xsub_joint_only_sgcn/test/2021-03-14T00-25-11/epoch29_test_score.pkl'
    ntu_xsub_jnt_orig = open_result_file(ntu_xsub_jnt_orig)

    ntu_xsub_bon_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/21-03-14-ntu_xsub_bone_only_sgcn/2021-03-14T09-25-25/epoch29_test_score.pkl'
    ntu_xsub_bon_orig = open_result_file(ntu_xsub_bon_orig)


    # ntu, xview
    ## DCT
    ntu_xset_jnt_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-07-ntu_xview_jnt_nerf_2_1ht_only_sgcn/2021-03-07T21-47-34/epoch48_test_score.pkl'
    ntu_xset_jnt_dct = open_result_file(ntu_xset_jnt_dct)

    ntu_xset_bon_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-07-ntu_xview_bon_nerf_2_1ht_only_sgcn/2021-03-07T21-51-32/epoch43_test_score.pkl'
    ntu_xset_bon_dct = open_result_file(ntu_xset_bon_dct)

    ## Nerf
    ntu_xset_jnt_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-10-ntu_xview_jnt_nerf_orig_2_1ht_only_sgcn/2021-03-10T20-59-48/epoch44_test_score.pkl'
    ntu_xset_jnt_nerf = open_result_file(ntu_xset_jnt_nerf)

    ntu_xset_bon_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-10-ntu_xview_bon_nerf_orig_2_1ht_only_sgcn/2021-03-10T20-58-27/epoch52_test_score.pkl'
    ntu_xset_bon_nerf = open_result_file(ntu_xset_bon_nerf)

    ## Original
    ntu_xset_jnt_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/21-03-14-ntu_xset_jnt_only_sgcn/2021-03-14T15-44-58/epoch30_test_score.pkl'
    ntu_xset_jnt_orig = open_result_file(ntu_xset_jnt_orig)
    ntu_xset_bon_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/work_dir/21-03-14-ntu_xset_bon_only_sgcn/2021-03-15T00-25-24/epoch30_test_score.pkl'
    ntu_xset_bon_orig = open_result_file(ntu_xset_bon_orig)

    ## MSG3D
    ntu_xset_jnt_msg3d = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-11-ntu_xview_jnt_nerf_dct_2_1ht_msg3d/2021-03-11T18-21-24/epoch42_test_score.pkl'
    ntu_xset_jnt_msg3d = open_result_file(ntu_xset_jnt_msg3d)

    ntu_xset_bon_msg3d = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-11-ntu_xview_bon_nerf_dct_2_1ht_msg3d/2021-03-12T17-25-45/epoch45_test_score.pkl'
    ntu_xset_bon_msg3d = open_result_file(ntu_xset_bon_msg3d)

    msg3d_ntu_xset_jnt_ste = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-16-ntu60_xview_joint_2020_cvpr_msg3d_ste_k_3/2021-09-18T07-38-14/epoch60_test_score.pkl'
    )
    msg3d_ntu_xset_jnt_tte = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-16-ntu60_xview_joint_2020_cvpr_msg3d_tte_k_3/2021-09-18T07-34-51/epoch50_test_score.pkl'
    )
    msg3d_ntu_xset_bon_ste = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-16-ntu60_xview_bone_2020_cvpr_msg3d_ste_k_3/2021-09-18T18-15-11/epoch50_test_score.pkl'
    )
    msg3d_ntu_xset_bon_tte = open_result_file(
        '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-16-ntu60_xview_bone_2020_cvpr_msg3d_tte_k_3/2021-09-18T07-52-44/epoch50_test_score.pkl'
    )

    # Kinetics
    ## DCT
    kinectics_dct_jnt = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-08-kinetics_jnt_nerf_2_1ht_only_sgcn/2021-03-08T07-00-32/epoch46_test_score.pkl'
    kinectics_dct_jnt = open_result_file(kinectics_dct_jnt)

    kinectics_dct_bon = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-08-kinetics_bon_nerf_2_1ht_only_sgcn/2021-03-08T08-15-46/epoch49_test_score.pkl'
    kinectics_dct_bon = open_result_file(kinectics_dct_bon)

    kinectics_bon_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-03-13-kinetics_bon_nerf_orig_2_1ht_only_sgcn/2021-03-12T22-17-37/epoch49_test_score.pkl'
    kinectics_bon_nerf = open_result_file(kinectics_bon_nerf)

    kin_jnt_bon_ang_arms_legs_head_21_01_06 = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-06-kinetics_joint_bone_angle_cangle_arms_legs_head_only_sgcn/2021-01-06T21-51-04/epoch45_test_score.pkl'
    kin_jnt_bon_ang_arms_legs_head_21_01_06 = open_result_file(kin_jnt_bon_ang_arms_legs_head_21_01_06)


    # Error analysis
    correct_dict = defaultdict(list)
    true_labels = []
    predicted_labels = []

    # 我想visualize的错误结果
    monitored_wrong_ones = []

    # correct and wrong files
    f_r = open(os.path.join('analysis', 'right_file.txt'), 'w')
    f_w = open(os.path.join('analysis', 'wrong_file.txt'), 'w')

    score_list = []

    # correct analysis
    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label[0]))):
        rgb_idx, l = label[:, i]
        if data_type == 'ntu120_xsub':
            _, ntu120_xsub_jnt_dct_ = ntu120_xsub_jnt_dct[i]
            _, ntu120_xsub_bon_dct_ = ntu120_xsub_bon_dct[i]
            _, ntu120_xsub_jnt_vel_dct_ = ntu120_xsub_jnt_vel_dct[i]
            _, ntu120_xsub_bon_vel_dct_ = ntu120_xsub_bon_vel_dct[i]

            _, ntu120_xsub_jnt_nerf_ = ntu120_xsub_jnt_nerf[i]
            _, ntu120_xsub_bon_nerf_ = ntu120_xsub_bon_nerf[i]

            _, ntu120_xsub_jnt_1ht_ = ntu120_xsub_jnt_1ht[i]
            _, ntu120_xsub_bon_1ht_ = ntu120_xsub_bon_1ht[i]

            _, ntu120_xsub_jnt_ste_msg3d_ = ntu120_xsub_jnt_ste_msg3d[i]
            _, ntu120_xsub_jnt_tte_msg3d_ = ntu120_xsub_jnt_tte_msg3d[i]
            _, ntu120_xsub_bon_ste_msg3d_ = ntu120_xsub_bon_ste_msg3d[i]
            _, ntu120_xsub_bon_tte_msg3d_ = ntu120_xsub_bon_tte_msg3d[i]

            # r = ntu120_xsub_jnt_dct_ + ntu120_xsub_bon_dct_ + \
            #     ntu120_xsub_jnt_vel_dct_ + ntu120_xsub_bon_vel_dct_ + \
            #     ntu120_xsub_jnt_nerf_ + ntu120_xsub_bon_nerf_
            # r = ntu120_xsub_jnt_dct_ + ntu120_xsub_bon_dct_ \
            #     + ntu120_xsub_jnt_nerf_ + ntu120_xsub_bon_nerf_
            r = ntu120_xsub_jnt_tte_msg3d_ + ntu120_xsub_bon_tte_msg3d_

        elif data_type == 'ntu120_xset':
            _, ntu120_xset_jnt_dct_ = ntu120_xset_jnt_dct[i]
            _, ntu120_xset_bon_dct_ = ntu120_xset_bon_dct[i]

            _, ntu120_xset_jnt_nerf_ = ntu120_xset_jnt_nerf[i]
            _, ntu120_xset_bon_nerf_ = ntu120_xset_bon_nerf[i]

            _, ntu120_xset_jnt_orig_ = ntu120_xset_jnt_orig[i]
            _, ntu120_xset_bon_orig_ = ntu120_xset_bon_orig[i]

            _, msg3d_ntu120_xset_joint_tte_k_3_ = msg3d_ntu120_xset_joint_tte_k_3[i]
            _, msg3d_ntu120_xset_bone_tte_k_3_ = msg3d_ntu120_xset_bone_tte_k_3[i]

            # r = ntu120_xset_jnt_dct_ + ntu120_xset_bon_dct_ \
            #     + ntu120_xset_jnt_nerf_ + ntu120_xset_bon_nerf_
            r = msg3d_ntu120_xset_joint_tte_k_3_ + msg3d_ntu120_xset_bone_tte_k_3_

        elif data_type == 'kinetics':
            _, kinectics_dct_jnt_ = kinectics_dct_jnt[i]
            _, kinectics_dct_bon_ = kinectics_dct_bon[i]
            _, kin_jnt_bon_ang_arms_legs_head_21_01_06_ = kin_jnt_bon_ang_arms_legs_head_21_01_06[i]

            _, kinectics_nerf_bon_ = kinectics_bon_nerf[i]

            r = kinectics_dct_jnt_ + kinectics_dct_bon_ + kinectics_nerf_bon_ + kin_jnt_bon_ang_arms_legs_head_21_01_06_

        elif data_type == 'ntu_xsub':
            _, ntu_xsub_jnt_dct_ = ntu_xsub_jnt_dct[i]
            _, ntu_xsub_bon_dct_ = ntu_xsub_bon_dct[i]
            _, ntu_xsub_jnt_nerf_ = ntu_xsub_jnt_nerf[i]
            _, ntu_xsub_bon_nerf_ = ntu_xsub_bon_nerf[i]

            _, ntu_xsub_jnt_dct_msg3d_ = ntu_xsub_jnt_dct_msg3d[i]
            _, ntu_xsub_bon_dct_msg3d_ = ntu_xsub_bon_dct_msg3d[i]

            _, ntu_xsub_jnt_orig_ = ntu_xsub_jnt_orig[i]
            _, ntu_xsub_bon_orig_ = ntu_xsub_bon_orig[i]

            # r = ntu_xsub_jnt_dct_ + ntu_xsub_bon_dct_ \
            #     + ntu_xsub_jnt_nerf_ + ntu_xsub_bon_nerf_
            # r = ntu_xsub_jnt_dct_msg3d_ + ntu_xsub_bon_dct_msg3d_
            # r = ntu_xsub_jnt_nerf_ + ntu_xsub_bon_nerf_ + ntu_xsub_jnt_dct_ + ntu_xsub_bon_dct_
            # r = 1.1 * ntu_xsub_jnt_dct_ + 1.05 * ntu_xsub_bon_dct_
            r = ntu_xsub_jnt_orig_

        elif data_type == 'ntu_xview':
            _, ntu_xset_jnt_dct_ = ntu_xset_jnt_dct[i]
            _, ntu_xset_bon_dct_ = ntu_xset_bon_dct[i]

            _, ntu_xset_jnt_nerf_ = ntu_xset_jnt_nerf[i]
            _, ntu_xset_bon_nerf_ = ntu_xset_bon_nerf[i]

            _, ntu_xset_jnt_msg3d_ = ntu_xset_jnt_msg3d[i]
            _, ntu_xset_bon_msg3d_ = ntu_xset_bon_msg3d[i]

            _, ntu_xset_jnt_orig_ = ntu_xset_jnt_orig[i]
            _, ntu_xset_bon_orig_ = ntu_xset_bon_orig[i]

            _, msg3d_ntu_xset_jnt_ste_ = msg3d_ntu_xset_jnt_ste[i]
            _, msg3d_ntu_xset_jnt_tte_ = msg3d_ntu_xset_jnt_tte[i]
            _, msg3d_ntu_xset_bon_ste_ = msg3d_ntu_xset_bon_ste[i]
            _, msg3d_ntu_xset_bon_tte_ = msg3d_ntu_xset_bon_tte[i]

            # r = ntu_xset_jnt_dct_ + ntu_xset_bon_dct_ \
            #     + ntu_xset_jnt_nerf_ + ntu_xset_bon_nerf_

            r = msg3d_ntu_xset_jnt_tte_ + msg3d_ntu_xset_jnt_ste_

        correct_dict[l].append(int(np.argmax(r) == int(l)))

        softmax = torch.softmax(torch.tensor(r), dim=-1)

        if int(l) == 72:
            score_list.append(softmax[int(l)].item())

        rank_5 = r.argsort()[-5:]

        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)

        # if int(l) == 85 and r == 85: # r == 82
        # if int(l) == 82-1 and r == 83-1: # r == 82
        #     print('i: ', i, 'rgb_idx: ', rgb_idx)

        right_num += int(r == int(l))
        if r == int(l):
            f_r.write(str(str(i) + ',' + str(r) + ',' + str(l) + '\n'))
        else:
            f_w.write(str(str(i) + ',' + str(r) + ',' + str(l) + '\n'))

        true_labels.append(int(l))
        predicted_labels.append(r)

        total_num += 1

    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    print('scores: ', score_list)

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