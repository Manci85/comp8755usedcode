import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import math
from numpy import linalg as LA

import time

# tmp = [[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]]
#
# tmp_2 = [[1, 3, 45],
#          [4, 5, 6],
#          [7, 8, 9]]
#
# tmp_ = LA.norm(tmp, axis=-1, keepdims=True)
# print('tmp_: ', tmp_)
# print('divided tmp: ', tmp / tmp_)
# tmp_2_ = LA.norm(tmp_2, axis=-1, keepdims=True)
# rst = (tmp / tmp_) * (tmp_2 / tmp_2_)
# rst = 1.0 - np.sum(rst, axis=-1)
# print('tmp: ', rst)
#
# assert 0


def angle(v1, v2):
    v1_n = v1 / LA.norm(v1, axis=1, keepdims=True)
    v2_n = v2 / LA.norm(v2, axis=1, keepdims=True)
    dot_v1_v2 = v1_n * v2_n
    dot_v1_v2 = 1.0 - np.sum(dot_v1_v2, axis=1)
    dot_v1_v2 = np.nan_to_num(dot_v1_v2)
    return dot_v1_v2


ntu_belly = 2

ntu_skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
)

ntu_skeleton_orig_bone_pairs = {
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

ntu_bone_adj = {
    25: 12,
    24: 12,
    12: 11,
    11: 10,
    10: 9,
    9: 21,
    21: 21,
    5: 21,
    6: 5,
    7: 6,
    8: 7,
    22: 8,
    23: 8,
    3: 21,
    4: 3,
    2: 21,
    1: 2,
    17: 1,
    18: 17,
    19: 18,
    20: 19,
    13: 1,
    14: 13,
    15: 14,
    16: 15
}

ntu_bone_angle_pairs = {
    25: (24, 12),
    24: (25, 12),
    12: (24, 25),
    11: (12, 10),
    10: (11, 9),
    9: (10, 21),
    21: (9, 5),
    5: (21, 6),
    6: (5, 7),
    7: (6, 8),
    8: (23, 22),
    22: (8, 23),
    23: (8, 22),
    3: (4, 21),
    4: (4, 4),
    2: (21, 1),
    1: (17, 13),
    17: (18, 1),
    18: (19, 17),
    19: (20, 18),
    20: (20, 20),
    13: (1, 14),
    14: (13, 15),
    15: (14, 16),
    16: (16, 16)
}

bone_pairs = {
    'ntu/xview': ntu_skeleton_bone_pairs,
    'ntu/xsub': ntu_skeleton_bone_pairs,

    # NTU 120 uses the same skeleton structure as NTU 60
    'ntu120/xsub': ntu_skeleton_bone_pairs,
    'ntu120/xset': ntu_skeleton_bone_pairs,

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
    parser.add_argument('--dataset', choices=['ntu', 'ntu120', 'kinetics'])
    parser.add_argument('--edge-type', choices=['joint_angle'])
    args = parser.parse_args()
    args.dataset = 'ntu'
    args.edge_type = 'joint_bone_angle_cangle_arms_legs'

    save_name = None

    if args.edge_type == 'joint_angle':
        save_name = '../data/{}/{}_data_joint_angle.npy'
    elif args.edge_type == 'joint_bone_angle':
        save_name = '../data/{}/{}_data_joint_bone_angle.npy'
    elif args.edge_type == 'bone_angle':
        save_name = '../data/{}/{}_data_bone_angle.npy'
    elif args.edge_type == 'joint_bone_angle_cangle':
        save_name = '../data/{}/{}_data_joint_bone_angle_cangle.npy'
    elif args.edge_type == 'joint_bone_angle_cangle_cangle2':
        save_name = '../data/{}/{}_data_joint_bone_angle_cangle_cangle2.npy'
    elif args.edge_type == 'joint_bone_angle_cangle_hands':
        save_name = '../data/{}/{}_data_joint_bone_angle_cangle_hands.npy'
    elif args.edge_type == 'joint_bone_angle_cangle_hands_2hands':
        save_name = '../data/{}/{}_data_joint_bone_angle_cangle_hands_2hands.npy'
    elif args.edge_type == 'joint_bone_angle_cangle_hands_angle_adj':
        save_name = '../data/{}/{}_data_joint_bone_angle_cangle_hands_angle_adj.npy'
    elif args.edge_type == 'joint_bone_angle_cangle_arms_legs':
        save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data/' \
                    '{}/{}_data_joint_bone_angle_cangle_arms_legs.npy'
    else:
        raise NotImplementedError('Unsupported edge type. ')

    print('save name: ', save_name)

    if args.edge_type == 'joint_angle':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                # data = data[:100]

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 4, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                # 我想调查为什么第二个人的feature全是0
                # a_data_idx = 0
                # for a_data in data:
                #     print(a_data[:, 0, 0, 1])
                #     print('a data idx: ', a_data_idx)
                #     a_data_idx += 1

                angle_list = []
                print('start --- ')
                for a_key, a_angle_value in tqdm(ntu_bone_angle_pairs.items()):
                    the_joint = a_key - 1
                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = fp_sp[:, :3, :, v1, :] - fp_sp[:, :3, :, the_joint, :]
                    vec2 = fp_sp[:, :3, :, v2, :] - fp_sp[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 3, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)
                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 3, 0, the_joint, 0])

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))

    elif args.edge_type == 'joint_bone_angle':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 7, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(ntu_bone_angle_pairs):
                    a_angle_value = ntu_bone_angle_pairs[a_key]
                    a_bone_value = ntu_bone_adj[a_key]
                    the_joint = a_key - 1

                    a_adj = a_bone_value - 1
                    fp_sp[:, 3:6, :, the_joint, :] = \
                        data[:, :3, :, the_joint, :] - data[:, :3, :, a_adj, :]

                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = fp_sp[:, :3, :, v1, :] - fp_sp[:, :3, :, the_joint, :]
                    vec2 = fp_sp[:, :3, :, v2, :] - fp_sp[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, -1, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 3, 0, the_joint, 0])

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))

    elif args.edge_type == 'bone_angle':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 4, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                # fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(ntu_bone_angle_pairs):
                    a_angle_value = ntu_bone_angle_pairs[a_key]
                    a_bone_value = ntu_skeleton_orig_bone_pairs[a_key]
                    the_joint = a_key - 1

                    a_adj = a_bone_value - 1
                    fp_sp[:, 0:3, :, the_joint, :] = \
                        data[:, :3, :, the_joint, :] - data[:, :3, :, a_adj, :]

                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = fp_sp[:, :3, :, v1, :] - fp_sp[:, :3, :, the_joint, :]
                    vec2 = fp_sp[:, :3, :, v2, :] - fp_sp[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, -1, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 3, 0, the_joint, 0])

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))

    elif args.edge_type == 'joint_bone_angle_cangle':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 8, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(ntu_bone_angle_pairs):
                    a_angle_value = ntu_bone_angle_pairs[a_key]
                    a_bone_value = ntu_bone_adj[a_key]
                    the_joint = a_key - 1

                    # 骨头
                    a_adj = a_bone_value - 1
                    fp_sp[:, 3:6, :, the_joint, :] = \
                        data[:, :3, :, the_joint, :] - data[:, :3, :, a_adj, :]

                    # 骨头夹角
                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = data[:, :3, :, v1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, v2, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 6, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 6, 0, the_joint, 0])

                    # 身体夹角
                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = data[:, :3, :, 2-1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 21-1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, -1, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))

    elif args.edge_type == 'joint_bone_angle_cangle_cangle2':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data'
                               '/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 9, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(ntu_bone_angle_pairs):
                    a_angle_value = ntu_bone_angle_pairs[a_key]
                    a_bone_value = ntu_bone_adj[a_key]
                    the_joint = a_key - 1

                    # 骨头
                    a_adj = a_bone_value - 1
                    fp_sp[:, 3:6, :, the_joint, :] = \
                        data[:, :3, :, the_joint, :] - data[:, :3, :, a_adj, :]

                    # 骨头夹角
                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = data[:, :3, :, v1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, v2, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 6, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 6, 0, the_joint, 0])

                    # 身体夹角
                    vec1 = data[:, :3, :, 2-1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 21-1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 7, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 与2和21的夹角
                    vec1 = data[:, :3, :, the_joint, :] - data[:, :3, :, 21 - 1, :]
                    vec2 = data[:, :3, :, 2 - 1, :] - data[:, :3, :, 21 - 1, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 8, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))

    # 相对于手的量量
    elif args.edge_type == 'joint_bone_angle_cangle_hands':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 11, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(ntu_bone_angle_pairs):
                    a_angle_value = ntu_bone_angle_pairs[a_key]
                    a_bone_value = ntu_bone_adj[a_key]
                    the_joint = a_key - 1

                    # 骨头
                    a_adj = a_bone_value - 1
                    fp_sp[:, 3:6, :, the_joint, :] = \
                        data[:, :3, :, the_joint, :] - data[:, :3, :, a_adj, :]

                    # 骨头夹角
                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = data[:, :3, :, v1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, v2, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 6, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 6, 0, the_joint, 0])

                    # 身体夹角
                    vec1 = data[:, :3, :, 2-1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 21-1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 7, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 与2和21的夹角
                    vec1 = data[:, :3, :, the_joint, :] - data[:, :3, :, 21 - 1, :]
                    vec2 = data[:, :3, :, 2 - 1, :] - data[:, :3, :, 21 - 1, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 8, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和左边的手的夹角
                    vec1 = data[:, :3, :, 24 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 25 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 9, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和右边的手的夹角
                    vec1 = data[:, :3, :, 22 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 23 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 10, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))

    # 两个手
    elif args.edge_type == 'joint_bone_angle_cangle_hands_2hands':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 12, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(ntu_bone_angle_pairs):
                    a_angle_value = ntu_bone_angle_pairs[a_key]
                    a_bone_value = ntu_bone_adj[a_key]
                    the_joint = a_key - 1

                    # 骨头
                    a_adj = a_bone_value - 1
                    fp_sp[:, 3:6, :, the_joint, :] = \
                        data[:, :3, :, the_joint, :] - data[:, :3, :, a_adj, :]

                    # 骨头夹角
                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = data[:, :3, :, v1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, v2, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 6, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 6, 0, the_joint, 0])

                    # 身体夹角
                    vec1 = data[:, :3, :, 2-1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 21-1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 7, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 与2和21的夹角
                    vec1 = data[:, :3, :, the_joint, :] - data[:, :3, :, 21 - 1, :]
                    vec2 = data[:, :3, :, 2 - 1, :] - data[:, :3, :, 21 - 1, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 8, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和左边的手的夹角
                    vec1 = data[:, :3, :, 24 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 25 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 9, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和右边的手的夹角
                    vec1 = data[:, :3, :, 22 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 23 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 10, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和两个手的夹角
                    vec1 = data[:, :3, :, 24 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 22 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 11, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))

    # Angle Adj Matrix
    elif args.edge_type == 'joint_bone_angle_cangle_hands_angle_adj':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 36, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(ntu_bone_angle_pairs):
                    a_angle_value = ntu_bone_angle_pairs[a_key]
                    a_bone_value = ntu_bone_adj[a_key]
                    the_joint = a_key - 1

                    # 骨头
                    a_adj = a_bone_value - 1
                    fp_sp[:, 3:6, :, the_joint, :] = \
                        data[:, :3, :, the_joint, :] - data[:, :3, :, a_adj, :]

                    # 骨头夹角
                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = data[:, :3, :, v1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, v2, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 6, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 6, 0, the_joint, 0])

                    # 身体夹角
                    vec1 = data[:, :3, :, 2 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 21 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 7, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 与2和21的夹角
                    vec1 = data[:, :3, :, the_joint, :] - data[:, :3, :, 21 - 1, :]
                    vec2 = data[:, :3, :, 2 - 1, :] - data[:, :3, :, 21 - 1, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 8, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和左边的手的夹角
                    vec1 = data[:, :3, :, 24 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 25 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 9, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和右边的手的夹角
                    vec1 = data[:, :3, :, 22 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 23 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 10, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 角度的adj matrix
                    vec1 = data[:, :3, :, the_joint, :] - data[:, :3, :, 2 - 1, :]
                    for a_i in range(1, 25+1):
                        vec2 = data[:, :3, :, a_i-1, :] - data[:, :3, :, 2 - 1, :]
                        fp_sp[:, a_i + 10, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))

    # Angle between hands, ankles, knees, feet
    elif args.edge_type == 'joint_bone_angle_cangle_arms_legs':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/data'
                               '/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')

                # sum_2 = np.sum(data[:, :, :, :, 1])
                # print('sum 2: ', data[:, :, 0, 0, 1])

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 15, T, V, M))
                print('fp_sp: ', fp_sp.shape)

                fp_sp[:, :C, :, :, :] = data

                angle_list = []
                print('start --- ')
                for a_key in tqdm(ntu_bone_angle_pairs):
                    a_angle_value = ntu_bone_angle_pairs[a_key]
                    a_bone_value = ntu_bone_adj[a_key]
                    the_joint = a_key - 1

                    # 骨头
                    a_adj = a_bone_value - 1
                    fp_sp[:, 3:6, :, the_joint, :] = \
                        data[:, :3, :, the_joint, :] - data[:, :3, :, a_adj, :]

                    # 骨头夹角
                    v1 = a_angle_value[0] - 1
                    v2 = a_angle_value[1] - 1
                    vec1 = data[:, :3, :, v1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, v2, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 6, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # print('fp sp a sample: ', fp_sp[0, 3, 0, the_joint, 0])
                    angle_list.append(fp_sp[0, 6, 0, the_joint, 0])

                    # 身体夹角
                    vec1 = data[:, :3, :, 2 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 21 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 7, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 与2和21的夹角
                    vec1 = data[:, :3, :, the_joint, :] - data[:, :3, :, 21 - 1, :]
                    vec2 = data[:, :3, :, 2 - 1, :] - data[:, :3, :, 21 - 1, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 8, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和左边的手的夹角
                    vec1 = data[:, :3, :, 24 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 25 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 9, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和右边的手的夹角
                    vec1 = data[:, :3, :, 22 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 23 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 10, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和两个手的夹角
                    vec1 = data[:, :3, :, 24 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 22 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 11, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和两个ankles的夹角
                    vec1 = data[:, :3, :, 10 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 6 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 12, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和两个knees的夹角
                    vec1 = data[:, :3, :, 18 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 14 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 13, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

                    # 和两个feet的夹角
                    vec1 = data[:, :3, :, 20 - 1, :] - data[:, :3, :, the_joint, :]
                    vec2 = data[:, :3, :, 16 - 1, :] - data[:, :3, :, the_joint, :]
                    # print('vec1: ', vec1.shape)
                    # print('vec2: ', vec2.shape)
                    fp_sp[:, 14, :, the_joint, :] = np.clip(angle(vec1, vec2), 0, 2)

        print('max angle list: ', np.max(angle_list))
        print('min angle list: ', np.min(angle_list))
