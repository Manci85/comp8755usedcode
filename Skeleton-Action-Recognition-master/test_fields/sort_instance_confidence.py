from ensemble_angular import open_result_file
import pickle
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    label_path_ntu_xsub = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/ntu/xsub/val_label.pkl'

    the_label = label_path_ntu_xsub

    with open(the_label, 'rb') as label:
        label = np.array(pickle.load(label))

    a_score_path = '/media/zhenyue-qin/Backup Plus/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/results_dw_beijing/21-07-29-ntu_xsub/2021-07-29T12-10-38/epoch57_test_score.pkl'
    a_score_f = open_result_file(a_score_path)

    confidences = []
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        _, predicted = a_score_f[i]
        confidences.append(predicted[int(l)])

    print('confidences: ', list(np.argsort(confidences)[:10]))
