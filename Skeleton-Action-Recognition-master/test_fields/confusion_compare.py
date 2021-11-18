import json


def read_json_as_dict(a_path):
    with open(a_path) as json_file:
        data = json.load(json_file)
        return data


acc_mat_jnt = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/IJCAI-Results/Confusion-Matrices/ntu120_xsub_joint/accuracy_per_class.json'
acc_mat_jnt = read_json_as_dict(acc_mat_jnt)
acc_mat_ang = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/IJCAI-Results/Confusion-Matrices/ntu120_xsub_angle/accuracy_per_class.json'
acc_mat_ang = read_json_as_dict(acc_mat_ang)
conf_mat_jnt = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/IJCAI-Results/Confusion-Matrices/ntu120_xsub_joint/confusion_matrix.json'
conf_mat_jnt = read_json_as_dict(conf_mat_jnt)
conf_mat_ang = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/IJCAI-Results/Confusion-Matrices/ntu120_xsub_angle/confusion_matrix.json'
conf_mat_ang = read_json_as_dict(conf_mat_ang)

for a_key in acc_mat_jnt:
    if acc_mat_jnt[a_key] < acc_mat_ang[a_key]:
        print(f'{a_key};\t\t\t\t '
              f'jnt: {acc_mat_jnt[a_key]};\t '
              f'ang: {acc_mat_ang[a_key]}\t'
              f'diff: {acc_mat_ang[a_key] - acc_mat_jnt[a_key]}\t'
              f'jnt confused: {conf_mat_jnt[a_key][:2]} '
              f'{float(conf_mat_jnt[a_key][0][-2:]) / float(conf_mat_jnt[a_key][2])}\t'
              f'ang confused: {conf_mat_ang[a_key][:2]} '
              f'{float(conf_mat_ang[a_key][0][-2:]) / float(conf_mat_ang[a_key][2])}\t')
