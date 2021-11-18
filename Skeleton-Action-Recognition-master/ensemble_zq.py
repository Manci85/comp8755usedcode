import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    alpha = 1

    label_path = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu120/xsub/val_label.pkl'

    with open(label_path, 'rb') as label:
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

    r_list = [
        r_5_5_5,     # 0
        r_10_10_5,   # 1
        r_15_10_5,   # 2
        r_15_15_5,   # 3
        r_20_10_5,   # 4
        r_20_15_5,   # 5
        r_20_15_15,  # 6
        r7,          # 7
        r_60_30_15,  # 8
        r_100_50_25, # 9
        r_150_50_25] # 10
    max_acc = 0
    for i_1 in range(len(r_list)-5):
        for i_2 in range(i_1+1, len(r_list)-4):
            for i_3 in range(i_2+1, len(r_list)-3):
                for i_4 in range(i_3+1, len(r_list)-2):
                    for i_5 in range(i_4+1, len(r_list)-1):
                        for i_6 in range(i_5+1, len(r_list)):
                            r = 0
                            print('\ni_1: ', i_1, '\ti_2: ', i_2, '\ti_3: ', i_3,
                                  '\ti_4: ', i_4, '\ti_5: ', i_5, '\ti_6: ', i_6)
                            r_list_tgt = [r_list[i_1], r_list[i_2], r_list[i_3],
                                          r_list[i_4], r_list[i_5], r_list[i_6]]

                            right_num = total_num = right_num_5 = 0
                            for i in range(len(label[0])):
                                _, l = label[:, i]

                                _, r1010 = r_10[i]
                                _, r1111 = r_11[i]
                                _, r55 = r5[i]

                                for a_r in r_list_tgt:
                                    _, a_r_ = a_r[i]
                                    r += a_r_
                                r = 1.1 * r + 1.0 * (r1010 + r1111)
                                # r = r

                                rank_5 = r.argsort()[-5:]
                                right_num_5 += int(int(l) in rank_5)
                                r = np.argmax(r)
                                right_num += int(r == int(l))
                                total_num += 1
                            acc = right_num / total_num
                            acc5 = right_num_5 / total_num

                            print('Top1 Acc: {:.4f}%'.format(acc * 100))
                            print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

                            if max_acc < acc:
                                max_acc = acc
    print('max acc: ', max_acc)
