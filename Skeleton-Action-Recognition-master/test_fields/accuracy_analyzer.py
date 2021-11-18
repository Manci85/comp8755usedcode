import json
import csv

def read_json_as_dict(a_path):
    with open(a_path) as json_file:
        data = json.load(json_file)
        return data

acc_dict_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-04-09-ntu120_xsub_jnt_only_sgcn/2021-04-09T07-21-15/accuracy_info/epoch_42_test_accuracy_per_class.json'
acc_dict_orig = read_json_as_dict(acc_dict_orig)
mat_dict_orig = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-04-09-ntu120_xsub_jnt_only_sgcn/2021-04-09T07-21-15/accuracy_info/epoch_42_test_confusion_matrix.json'
mat_dict_orig = read_json_as_dict(mat_dict_orig)

acc_dict_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-04-07-ntu120_xsub_jnt_dct_cat_3_1ht_only_sgcn/2021-04-08T09-46-31/accuracy_info/epoch_60_test_accuracy_per_class.json'
acc_dict_nerf = read_json_as_dict(acc_dict_nerf)
mat_dict_nerf = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-04-07-ntu120_xsub_jnt_dct_cat_3_1ht_only_sgcn/2021-04-08T09-46-31/accuracy_info/epoch_60_test_confusion_matrix.json'
mat_dict_nerf = read_json_as_dict(mat_dict_nerf)

acc_dict_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-22-ntu120_xsub_joint_2021_qin_msgcn_chron_0p1/2021-09-23T22-59-57/accuracy_info/epoch_100_test_accuracy_per_class.json'
acc_dict_dct = read_json_as_dict(acc_dict_dct)
mat_dict_dct = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-09-22-ntu120_xsub_joint_2021_qin_msgcn_chron_0p1/2021-09-23T22-59-57/accuracy_info/epoch_100_test_confusion_matrix.json'
mat_dict_dct = read_json_as_dict(mat_dict_dct)

nerf_imp_num = 0
dct_imp_num = 0

type_1 = 'ntu120_joint_tte'
type_2 = 'chron'

a_csv = open(f'analysis/{type_1}_{type_2}_acc_analyse.csv', 'w')
a_csv.write(f"Action,Orig Acc,Orig Confusion,{type_1} Acc,{type_1} Confusion,{type_1} Improvement,{type_2} Acc,{type_2} Confusion,{type_2} Improvement\n")
for a_key, a_value in acc_dict_orig.items():
    # if acc_dict_orig[a_key] <= acc_dict_nerf[a_key]:
    if acc_dict_orig[a_key] > acc_dict_nerf[a_key]:
        nerf_imp_num += 1
    # if acc_dict_orig[a_key] <= acc_dict_dct[a_key]:
    if acc_dict_orig[a_key] > acc_dict_dct[a_key]:
        dct_imp_num += 1
    # if acc_dict_orig[a_key] < acc_dict_nerf[a_key] and acc_dict_orig[a_key] < acc_dict_dct[a_key]:
    # if acc_dict_orig[a_key] < acc_dict_nerf[a_key] and acc_dict_orig[a_key] < acc_dict_dct[a_key]:
    nerf_imp = format(acc_dict_nerf[a_key]-acc_dict_orig[a_key], '.4f')
    dct_imp = format(acc_dict_dct[a_key]-acc_dict_orig[a_key], '.4f')
    a_line = f"{a_key}," \
             f"{acc_dict_orig[a_key]},{mat_dict_orig[a_key][1]}," \
             f"{acc_dict_nerf[a_key]},{mat_dict_nerf[a_key][1]},{nerf_imp}," \
             f"{acc_dict_dct[a_key]},{mat_dict_dct[a_key][1]},{dct_imp}\n"
    a_csv.write(a_line)

print(f'# {type_1} imp: ', nerf_imp_num, f'# {type_2} imp: ', dct_imp_num)
