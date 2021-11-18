import sys
sys.path.extend(['../'])

import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm

from data_gen.preprocess import pre_normalization
import numpy as np


# NTU RGB+D Skeleton 120 Configurations: https://arxiv.org/pdf/1905.04757.pdf
training_subjects = set([
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
])

# Even numbered setups (2,4,...,32) used for training
training_setups = set(range(2, 33, 2))

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300


def read_skeleton_filter(path):
    with open(path, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(path, max_body=4, num_joint=25):
    seq_info = read_skeleton_filter(path)
    # Create single skeleton tensor: (M, T, V, C)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    # To (C,T,V,M)
    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(file_list, out_path, ignored_sample_path, part):
    existing_name = 'jnt_bon_ang_arms_legs_cuda'
    existing_path = '{}/{}_data_{}.npy'.format(out_path, part, existing_name)
    print('Loading existing path: ', existing_path)
    if os.path.exists(existing_path):
        fp = np.load(existing_path)

        data_cp = fp.copy()
        for i in range(1, fp.shape[2]):
            fp[:, :, i, :, :] = data_cp[:, :, i, :, :] - data_cp[:, :, i - 1, :, :]
        fp[:, :, 0, :, :] = np.zeros_like(fp[:, :, 0, :, :])
    else:
        print('existing path loading failed')
        raise NotImplementedError
    np.save('{}/{}_data_{}_velocity.npy'.format(out_path, part, existing_name), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D 120 Skeleton Data Extraction')
    parser.add_argument('--ignored-sample-path',
                        default='../../../Datasets/nturgbd_raw/NTU_RGBD120_samples_with_missing_skeletons.txt')
    parser.add_argument('--out-folder',
                        default=
                        '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/kinetics/')

    part = ['train', 'val']
    arg = parser.parse_args()

    # Combine skeleton file paths
    # file_list = []
    # for folder in [arg.part1_path, arg.part2_path]:
    #     for path in os.listdir(folder):
    #         file_list.append((folder, path))
    file_list = None

    print('out folder: ', arg.out_folder)
    for p in part:
        out_path = arg.out_folder
        print(p)
        gendata(file_list, out_path, arg.ignored_sample_path, part=p)

