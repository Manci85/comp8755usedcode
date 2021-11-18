import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from numpy import linalg as LA
import torch.nn as nn
import torch

import time


# Hyperparameters
bch_sz = 2000
print_freq = 3


def angle(v1, v2):
    v1_n = v1 / LA.norm(v1, axis=1, keepdims=True)
    v2_n = v2 / LA.norm(v2, axis=1, keepdims=True)
    dot_v1_v2 = v1_n * v2_n
    dot_v1_v2 = 1.0 - np.sum(dot_v1_v2, axis=1)
    dot_v1_v2 = np.nan_to_num(dot_v1_v2)
    return dot_v1_v2


kinetics_bone_adj = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 1,
    6: 5,
    7: 6,
    8: 2,
    9: 8,
    10: 9,
    11: 5,
    12: 11,
    13: 12,
    14: 0,
    15: 0,
    16: 14,
    17: 15
}


kinetics_bone_angle_pairs = {
    0: (3, 6),
    1: (2, 5),
    2: (3, 8),
    3: (4, 2),
    4: (4, 4),
    5: (11, 6),
    6: (5, 7),
    7: (7, 7),
    8: (2, 9),
    9: (8, 10),
    10: (10, 10),
    11: (5, 12),
    12: (11, 13),
    13: (13, 13),
    14: (3, 4),
    15: (6, 7),
    16: (3, 4),
    17: (6, 7)
}


benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    # 'ntu120': ('ntu120/xset', 'ntu120/xsub'),
    'ntu120': ('ntu120/xsub',),
    'kinetics': ('kinetics',)
}

parts = {'train', 'val'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NTU60/NTU120/Kinetics')
    parser.add_argument('--dataset', choices=['ntu', 'ntu120', 'kinetics'])
    parser.add_argument('--edge-type')
    args = parser.parse_args()
    args.dataset = 'kinetics'
    args.edge_type = 'joint_bone_angle_arms_legs'

    save_name = None

    if args.edge_type == 'joint_bone_angle_arms_legs':
        save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                    'data/{}/{}_data_jnt_bon_ang_arms_legs_cuda.npy'
        # save_name = '../data/{}/{}_data_jnt_bon_ang_arms_legs_cuda.npy'
    else:
        raise NotImplementedError('Unsupported edge type. ')

    print('save name: ', save_name)

    if args.edge_type == 'joint_bone_angle_arms_legs':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/'
                               'MS-G3D/data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                # data = np.load(
                #     '../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)

                # Number of batches
                bch_len = N // bch_sz

                print('creating fp sp...')

                # fp_sp = None
                feature_dim = 12
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, feature_dim, T, V, M))

                for i in range(bch_len+1):
                    if i % print_freq == 0:
                        print(f'{i} out of {bch_len}')
                    a_bch = torch.tensor(data[i * bch_sz:(i + 1) * bch_sz])

                    # generating bones
                    fp_sp_joint_list_bone = []
                    fp_sp_joint_list_bone_angle = []
                    fp_sp_joint_list_body_center_angle_1 = []
                    fp_sp_joint_list_body_center_angle_2 = []
                    fp_sp_left_hand_angle = []
                    fp_sp_right_hand_angle = []
                    fp_sp_two_hand_angle = []
                    fp_sp_two_elbow_angle = []
                    fp_sp_two_knee_angle = []
                    fp_sp_two_feet_angle = []

                    all_list = [
                        fp_sp_joint_list_bone, fp_sp_joint_list_bone_angle, fp_sp_joint_list_body_center_angle_1,
                        fp_sp_joint_list_body_center_angle_2,
                        fp_sp_two_hand_angle, fp_sp_two_elbow_angle, fp_sp_two_knee_angle,
                        fp_sp_two_feet_angle
                    ]

                    # cosine
                    cos = nn.CosineSimilarity(dim=1, eps=0)

                    for a_key in kinetics_bone_angle_pairs:
                        a_angle_value = kinetics_bone_angle_pairs[a_key]
                        a_bone_value = kinetics_bone_adj[a_key]
                        the_joint = a_key
                        a_bch = a_bch.to('cuda')

                        # bone
                        a_adj = a_bone_value
                        bone_diff = (a_bch[:, :2, :, the_joint, :] -
                                     a_bch[:, :2, :, a_adj, :]).unsqueeze(3).cpu()
                        fp_sp_joint_list_bone.append(bone_diff)

                        # bone angles
                        v1 = a_angle_value[0] - 1
                        v2 = a_angle_value[1] - 1
                        vec1 = a_bch[:, :2, :, v1, :] - a_bch[:, :2, :, the_joint, :]
                        vec2 = a_bch[:, :2, :, v2, :] - a_bch[:, :2, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_joint_list_bone_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # body angles 1
                        vec1 = a_bch[:, :2, :, 0, :] - a_bch[:, :2, :, the_joint, :]
                        vec2 = a_bch[:, :2, :, 1, :] - a_bch[:, :2, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_joint_list_body_center_angle_1.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # body angles 2
                        vec1 = a_bch[:, :2, :, the_joint, :] - a_bch[:, :2, :, 1, :]
                        vec2 = a_bch[:, :2, :, 0, :] - a_bch[:, :2, :, 1, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_joint_list_body_center_angle_2.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # two hand angle
                        vec1 = a_bch[:, :2, :, 4, :] - a_bch[:, :2, :, the_joint, :]
                        vec2 = a_bch[:, :2, :, 7, :] - a_bch[:, :2, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_two_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # two elbow angle
                        vec1 = a_bch[:, :2, :, 3, :] - a_bch[:, :2, :, the_joint, :]
                        vec2 = a_bch[:, :2, :, 6, :] - a_bch[:, :2, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_two_elbow_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # two knee angle
                        vec1 = a_bch[:, :2, :, 9, :] - a_bch[:, :2, :, the_joint, :]
                        vec2 = a_bch[:, :2, :, 12, :] - a_bch[:, :2, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_two_knee_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # two feet angle
                        vec1 = a_bch[:, :2, :, 10, :] - a_bch[:, :2, :, the_joint, :]
                        vec2 = a_bch[:, :2, :, 13, :] - a_bch[:, :2, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_two_feet_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                    for a_list_id in range(len(all_list)):
                        all_list[a_list_id] = torch.cat(all_list[a_list_id], dim=3)

                    all_list = torch.cat(all_list, dim=1)

                    # Joint features.
                    fp_sp[i * bch_sz:(i + 1) * bch_sz, :2, :, :, :] = a_bch.cpu().numpy()[:, :2, :, :, :]
                    # Bone and angle features.
                    fp_sp[i * bch_sz:(i + 1) * bch_sz, 2:feature_dim-1, :, :, :] = all_list.numpy()
                    fp_sp[i * bch_sz:(i + 1) * bch_sz, -1, :, :, :] = a_bch.cpu().numpy()[:, -1, :, :, :]

                print('fp sp: ', fp_sp.shape)
