import json
import csv
import numpy as np

def read_json_as_dict(a_path):
    with open(a_path) as json_file:
        data = json.load(json_file)
        return data

def remove_str_digits(a_str):
    return ''.join([i for i in a_str if not i.isdigit()])

# bsl acc: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-06-ntu120_xsub_jnt_only_sgcn/2021-05-06T23-42-13/accuracy_info/epoch_31_test_accuracy_per_class.json'
# bsl conf: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-06-ntu120_xsub_jnt_only_sgcn/2021-05-06T23-42-13/accuracy_info/epoch_31_test_confusion_matrix.json'
# bsl v acc: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_vel_only_sgcn/2021-05-09T12-38-38/accuracy_info/epoch_30_test_accuracy_per_class.json'
# bsl v conf: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_vel_only_sgcn/2021-05-09T12-38-38/accuracy_info/epoch_30_test_confusion_matrix.json'
# local acc: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-07-ntu120_xsub_jnt_local_only_sgcn/2021-05-08T14-57-41/accuracy_info/epoch_45_test_accuracy_per_class.json'
# local conf: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-07-ntu120_xsub_jnt_local_only_sgcn/2021-05-08T14-57-41/accuracy_info/epoch_45_test_confusion_matrix.json'
# local v acc: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_local_vel_only_sgcn/2021-05-09T23-51-00/accuracy_info/epoch_40_test_accuracy_per_class.json
# local v conf: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_local_vel_only_sgcn/2021-05-09T23-51-00/accuracy_info/epoch_40_test_confusion_matrix.json'
# center acc: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-08-ntu120_xsub_jnt_center_only_sgcn/2021-05-08T14-53-09/accuracy_info/epoch_55_test_accuracy_per_class.json'
# center conf: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-08-ntu120_xsub_jnt_center_only_sgcn/2021-05-08T14-53-09/accuracy_info/epoch_55_test_confusion_matrix.json'
# center v acc: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_center_vel_only_sgcn/2021-05-09T12-39-35/accuracy_info/epoch_45_test_accuracy_per_class.json'
# center v conf: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_center_vel_only_sgcn/2021-05-09T12-39-35/accuracy_info/epoch_45_test_confusion_matrix.json'
# part acc: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-07-ntu120_xsub_jnt_part_only_sgcn/2021-05-09T00-54-59/accuracy_info/epoch_53_test_accuracy_per_class.json'
# part conf: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-07-ntu120_xsub_jnt_part_only_sgcn/2021-05-09T00-54-59/accuracy_info/epoch_53_test_confusion_matrix.json'
# part acc v: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_part_vel_only_sgcn/2021-05-09T23-53-44/accuracy_info/epoch_55_test_accuracy_per_class.json'
# part conf v: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_part_vel_only_sgcn/2021-05-09T23-53-44/accuracy_info/epoch_55_test_confusion_matrix.json'
# finger acc: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-08-ntu120_xsub_jnt_finger_only_sgcn/2021-05-08T11-09-07/accuracy_info/epoch_40_test_accuracy_per_class.json'
# finger conf: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-08-ntu120_xsub_jnt_finger_only_sgcn/2021-05-08T11-09-07/accuracy_info/epoch_40_test_confusion_matrix.json'
# finger acc v: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_finger_vel_only_sgcn/2021-05-09T13-04-22/accuracy_info/epoch_40_test_accuracy_per_class.json'
# finger conf v: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_finger_vel_only_sgcn/2021-05-09T13-04-22/accuracy_info/epoch_40_test_confusion_matrix.json'
# all acc: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-03-ntu120_xsub_joint_angle_arms_legs_onehot_only_sgcn/2021-01-04T12-07-30/acc_info/accuracy_per_class.json'
# all conf: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-03-ntu120_xsub_joint_angle_arms_legs_onehot_only_sgcn/2021-01-04T12-07-30/acc_info/confusion_matrix.json'
# all acc v: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-11-ntu120_xsub_joint_angle_v_hands_2hands_onehot_only_sgcn/2021-01-11T09-57-33/accuracy_info/accuracy_per_class.json'
# all conf v: '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-11-ntu120_xsub_joint_angle_v_hands_2hands_onehot_only_sgcn/2021-01-11T09-57-33/accuracy_info/confusion_matrix.json'

acc_dict_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_vel_only_sgcn/2021-05-09T12-38-38/accuracy_info/epoch_30_test_accuracy_per_class.json'
acc_dict_orig = read_json_as_dict(acc_dict_orig)
mat_dict_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-05-09-ntu120_xsub_jnt_vel_only_sgcn/2021-05-09T12-38-38/accuracy_info/epoch_30_test_confusion_matrix.json'
mat_dict_orig = read_json_as_dict(mat_dict_orig)

acc_dict_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-11-ntu120_xsub_joint_angle_v_hands_2hands_onehot_only_sgcn/2021-01-11T09-57-33/accuracy_info/accuracy_per_class.json'
acc_dict_nerf = read_json_as_dict(acc_dict_nerf)
mat_dict_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-01-11-ntu120_xsub_joint_angle_v_hands_2hands_onehot_only_sgcn/2021-01-11T09-57-33/accuracy_info/confusion_matrix.json'
mat_dict_nerf = read_json_as_dict(mat_dict_nerf)


nerf_imp_num = 0
dct_imp_num = 0

type_1 = 'all_reduce'
type_2 = 'center_reduce'

out_str = ""

imp_val_list = []
line_list = []

boost_acc = -0.01
# boost_acc = -0.02
for a_key, a_value in acc_dict_orig.items():
    acc_dict_orig[a_key] += boost_acc
    # if acc_dict_orig[a_key] <= acc_dict_nerf[a_key]:
    if acc_dict_orig[a_key] > acc_dict_nerf[a_key]:
        nerf_imp_num += 1


    orig_val = "{:.2%}".format(acc_dict_orig[a_key])
    new_val = "{:.2%}".format(acc_dict_nerf[a_key])
    imp_val = "{:.2%}".format(acc_dict_nerf[a_key] - acc_dict_orig[a_key])
    # imp_val_list.append(acc_dict_nerf[a_key] - acc_dict_orig[a_key])
    imp_val_list.append(acc_dict_orig[a_key])

    a_new_line = f"{a_key} & {orig_val} & {remove_str_digits(mat_dict_orig[a_key][1])}" \
                 f" & {new_val} & {imp_val}" \
                 f" & {remove_str_digits(mat_dict_nerf[a_key][1])} \\\\"

    line_list.append(a_new_line)

# arg_max_list = np.argsort(imp_val_list)[::-1]
arg_max_list = np.argsort(imp_val_list)
for a_idx in arg_max_list[60:120]:
    a_line = line_list[a_idx]
    out_str += a_line.replace('%', '')
    out_str += '\n'

print(out_str)
