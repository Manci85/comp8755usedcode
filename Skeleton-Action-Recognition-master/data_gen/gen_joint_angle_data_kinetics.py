import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import math
from numpy import linalg as LA

import time


def angle(v1, v2):
    v1_n = v1 / LA.norm(v1, axis=1, keepdims=True)
    v2_n = v2 / LA.norm(v2, axis=1, keepdims=True)
    dot_v1_v2 = v1_n * v2_n
    dot_v1_v2 = 1.0 - np.sum(dot_v1_v2, axis=1)
    dot_v1_v2 = np.nan_to_num(dot_v1_v2)
    return dot_v1_v2


kinetics_original_bone_adj = {
    25: 12,
    24: 25,
    23: 8,
    21: 21,
    22: 23,
    20: 19,
    19: 18,
    18: 17,
    17: 1,
    16: 15,
    15: 14,
    14: 13,
    13: 1,
    12: 11,
    11: 10,
    10: 9,
    9: 21,
    8: 7,
    7: 6,
    6: 5,
    5: 21,
    4: 3,
    3: 21,
    2: 21,
    1: 2
}

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

bone_pairs = {
    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
        (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )
}

benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    'ntu120': ('ntu120/xset',),
    'kinetics': ('kinetics',)
}

parts = {'train', 'val'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NTU60/NTU120/Kinetics')
    parser.add_argument('--dataset', choices=['kinetics'])
    parser.add_argument('--edge-type')
    args = parser.parse_args()
    args.dataset = 'kinetics'
    args.edge_type = 'joint_bone_angle_cangle_arms_legs_heads'

    save_name = None

    data_path = '../data'
    # data_path = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/data'

    if args.edge_type == 'joint_bone_angle_cangle_arms_legs':
        save_name = '{}/{}/{}_data_joint_bone_angle_cangle_arms_legs_test.npy'
    elif args.edge_type == 'joint_bone_angle_cangle_arms_legs_heads':
        save_name = '{}/{}/{}_data_joint_bone_angle_cangle_arms_legs_heads.npy'
    else:
        raise NotImplementedError('Unsupported edge type. ')

    print('save name: ', save_name)

    # Angle between hands, ankles, knees, feet
    if args.edge_type == 'joint_bone_angle_cangle_arms_legs':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('{}/{}/{}_data_joint.npy'.format(data_path, benchmark, part),
                               mmap_mode='r')

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(data_path, benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 13, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(kinetics_bone_angle_pairs):
                    a_angle_value = kinetics_bone_angle_pairs[a_key]
                    a_bone_value = kinetics_bone_adj[a_key]
                    the_joint = a_key

                    # 骨头
                    a_adj = a_bone_value
                    fp_sp[:, 3:6, :, the_joint, :] = \
                        data[:, :3, :, the_joint, :] - data[:, :3, :, a_adj, :]

                    # 骨头夹角
                    v1 = a_angle_value[0]
                    v2 = a_angle_value[1]
                    vec1 = data[:, :3, :, v1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, v2, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 6, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 6, -1, the_joint, 0])

                    # 身体夹角
                    vec1 = data[:, :3, :, 0, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 7, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 与2和21的夹角
                    vec1 = data[:, :3, :, the_joint, :] - data[:, :3, :, 1, :]
                    vec2 = data[:, :3, :, 0, :] - data[:, :3, :, 1, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 8, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和两个手的夹角
                    vec1 = data[:, :3, :, 4, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 7, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 9, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和两个elbows的夹角
                    vec1 = data[:, :3, :, 3, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 6, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 10, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和两个knees的夹角
                    vec1 = data[:, :3, :, 9, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 12, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 11, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和两个feet的夹角
                    vec1 = data[:, :3, :, 10, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 13, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 12, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))

    elif args.edge_type == 'joint_bone_angle_cangle_arms_legs_heads':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('{}/{}/{}_data_joint.npy'.format(data_path, benchmark, part),
                               mmap_mode='r')
                # data = data[:100]
                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(data_path, benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 15, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(kinetics_bone_angle_pairs):
                    a_angle_value = kinetics_bone_angle_pairs[a_key]
                    a_bone_value = kinetics_bone_adj[a_key]
                    the_joint = a_key

                    # head, joint, left foot
                    vec1 = data[:, :3, :, 0, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 10, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 13, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # head, joint, right foot
                    vec1 = data[:, :3, :, 0, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 13, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 14, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))