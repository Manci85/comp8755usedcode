import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from numpy import linalg as LA
import torch.nn as nn
import torch

import torch.nn.functional as F

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
    args.dataset = 'ntu120'
    args.edge_type = 'jnt_bon'

    save_name = None

    if args.edge_type == 'joint_bone_angle_arms_legs':
        save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                    'data/{}/{}_data_jnt_bon_ang_arms_legs_cuda.npy'
    elif args.edge_type == 'jnt_bon_nor_loc':
        save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                    'data/{}/{}_data_jnt_bon_nor_loc_cuda.npy'
    elif args.edge_type == 'jnt_bon':
        save_name = '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                    'data/{}/{}_data_jnt_bon_cuda.npy'
    else:
        raise NotImplementedError('Unsupported edge type. ')

    print('save name: ', save_name)

    if args.edge_type == 'joint_bone_angle_arms_legs':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/'
                               'MS-G3D/data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)

                # Number of batches
                bch_len = N // bch_sz

                print('creating fp sp...')

                # fp_sp = None
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 15, T, V, M))

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
                        fp_sp_joint_list_body_center_angle_2, fp_sp_left_hand_angle, fp_sp_right_hand_angle,
                        fp_sp_two_hand_angle, fp_sp_two_elbow_angle, fp_sp_two_knee_angle,
                        fp_sp_two_feet_angle
                    ]

                    # cosine
                    cos = nn.CosineSimilarity(dim=1, eps=0)

                    for a_key in ntu_bone_angle_pairs:
                        a_angle_value = ntu_bone_angle_pairs[a_key]
                        a_bone_value = ntu_bone_adj[a_key]
                        the_joint = a_key - 1
                        a_adj = a_bone_value - 1
                        a_bch = a_bch.to('cuda')
                        bone_diff = (a_bch[:, :3, :, the_joint, :] -
                                     a_bch[:, :3, :, a_adj, :]).unsqueeze(3).cpu()
                        fp_sp_joint_list_bone.append(bone_diff)

                        # bone angles
                        v1 = a_angle_value[0] - 1
                        v2 = a_angle_value[1] - 1
                        vec1 = a_bch[:, :3, :, v1, :] - a_bch[:, :3, :, the_joint, :]
                        vec2 = a_bch[:, :3, :, v2, :] - a_bch[:, :3, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_joint_list_bone_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # body angles 1
                        vec1 = a_bch[:, :3, :, 2 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        vec2 = a_bch[:, :3, :, 21 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_joint_list_body_center_angle_1.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # body angles 2
                        vec1 = a_bch[:, :3, :, the_joint, :] - a_bch[:, :3, :, 21 - 1, :]
                        vec2 = a_bch[:, :3, :, 2 - 1, :] - a_bch[:, :3, :, 21 - 1, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_joint_list_body_center_angle_2.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # left hand angle
                        vec1 = a_bch[:, :3, :, 24 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        vec2 = a_bch[:, :3, :, 25 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_left_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # right hand angle
                        vec1 = a_bch[:, :3, :, 22 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        vec2 = a_bch[:, :3, :, 23 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_right_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # two hand angle
                        vec1 = a_bch[:, :3, :, 24 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        vec2 = a_bch[:, :3, :, 22 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_two_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # two elbow angle
                        vec1 = a_bch[:, :3, :, 10 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        vec2 = a_bch[:, :3, :, 6 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_two_elbow_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # two knee angle
                        vec1 = a_bch[:, :3, :, 18 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        vec2 = a_bch[:, :3, :, 14 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_two_knee_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                        # two feet angle
                        vec1 = a_bch[:, :3, :, 20 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        vec2 = a_bch[:, :3, :, 16 - 1, :] - a_bch[:, :3, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        angular_feature[angular_feature != angular_feature] = 0
                        fp_sp_two_feet_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                    for a_list_id in range(len(all_list)):
                        all_list[a_list_id] = torch.cat(all_list[a_list_id], dim=3)

                    all_list = torch.cat(all_list, dim=1)

                    # if fp_sp is None:
                    #     # fp_sp = all_list
                    #     fp_sp = all_list.numpy()
                    # else:
                    #     # fp_sp = torch.cat((fp_sp, all_list), dim=0)
                    #     fp_sp = np.concatenate((fp_sp, all_list), axis=0)

                    # Joint features.
                    fp_sp[i * bch_sz:(i + 1) * bch_sz, :3, :, :, :] = a_bch.cpu().numpy()
                    # Bone and angle features.
                    fp_sp[i * bch_sz:(i + 1) * bch_sz, 3:, :, :, :] = all_list.numpy()

                print('fp sp: ', fp_sp.shape)
                # save_f_name = save_name.format(benchmark, part)
                # with open(save_f_name, 'wb') as f:
                #     np.save(f, fp_sp)

    elif args.edge_type == 'jnt_bon_nor_loc':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load('/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/'
                               'MS-G3D/data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                # data = np.load(
                #     '../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                # data = data[:2000]

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)

                # Number of batches
                bch_len = N // bch_sz

                print('creating fp sp...')

                # fp_sp = None
                # fp_sp = open_memmap(
                #     save_name.format(benchmark, part),
                #     dtype='float32',
                #     mode='w+',
                #     shape=(N, 9, T, V, M))
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 9, T, V, M))

                for i in range(bch_len+1):
                    if i % print_freq == 0:
                        print(f'{i} out of {bch_len}')
                    a_bch = torch.tensor(data[i * bch_sz:(i + 1) * bch_sz])

                    # generating bones
                    fp_sp_joint_list_bone = []
                    fp_sp_joint_list_bone_angle = []

                    all_list = [
                        fp_sp_joint_list_bone, fp_sp_joint_list_bone_angle
                    ]

                    # cosine
                    cos = nn.CosineSimilarity(dim=1, eps=0)

                    for a_key in ntu_bone_angle_pairs:
                        a_angle_value = ntu_bone_angle_pairs[a_key]
                        a_bone_value = ntu_bone_adj[a_key]
                        the_joint = a_key - 1
                        a_adj = a_bone_value - 1
                        a_bch = a_bch.to('cuda')
                        bone_diff = (a_bch[:, :3, :, the_joint, :] -
                                     a_bch[:, :3, :, a_adj, :]).unsqueeze(3).cpu()
                        fp_sp_joint_list_bone.append(bone_diff)

                        # bone angles
                        v1 = a_angle_value[0] - 1
                        v2 = a_angle_value[1] - 1
                        vec1 = a_bch[:, :3, :, v1, :] - a_bch[:, :3, :, the_joint, :]
                        vec2 = a_bch[:, :3, :, v2, :] - a_bch[:, :3, :, the_joint, :]
                        angular_feature = (1.0 - cos(vec1, vec2))
                        nor_feature = F.normalize(torch.cross(vec1, vec2, dim=1),
                                                  p=2, dim=1)

                        # norm_tmp_1 = nor_feature[1, :, 1, 0]
                        # vec_tmp_1 = vec1[1, :, 1, 0]
                        # vec_tmp_2 = vec2[1, :, 1, 0]
                        # tmp_1 = torch.sum(vec_tmp_1 * norm_tmp_1)
                        # tmp_2 = torch.sum(vec_tmp_2 * norm_tmp_1)

                        nor_feature[nor_feature != nor_feature] = 0
                        to_append = nor_feature.unsqueeze(-2).cpu()
                        fp_sp_joint_list_bone_angle.append(to_append)

                    for a_list_id in range(len(all_list)):
                        all_list[a_list_id] = torch.cat(all_list[a_list_id], dim=3)

                    all_list = torch.cat(all_list, dim=1)

                    # if fp_sp is None:
                    #     # fp_sp = all_list
                    #     fp_sp = all_list.numpy()
                    # else:
                    #     # fp_sp = torch.cat((fp_sp, all_list), dim=0)
                    #     fp_sp = np.concatenate((fp_sp, all_list), axis=0)

                    # Joint features.
                    fp_sp[i * bch_sz:(i + 1) * bch_sz, :3, :, :, :] = a_bch.cpu().numpy()
                    # Bone and other features.
                    fp_sp[i * bch_sz:(i + 1) * bch_sz, 3:, :, :, :] = all_list.numpy()

                print('fp sp: ', fp_sp.shape)
                # save_f_name = save_name.format(benchmark, part)
                # with open(save_f_name, 'wb') as f:
                #     np.save(f, fp_sp)

    elif args.edge_type == 'jnt_bon':
        for benchmark in benchmarks[args.dataset]:
            for part in parts:
                print(benchmark, part)
                data = np.load(
                    '/data1/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/'
                    'MS-G3D/data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                # data = np.load(
                #     '../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                # data = data[:2000]

                N, C, T, V, M = data.shape
                print('data shape: ', data.shape)

                # Number of batches
                bch_len = N // bch_sz

                print('creating fp sp...')

                # fp_sp = None
                # fp_sp = open_memmap(
                #     save_name.format(benchmark, part),
                #     dtype='float32',
                #     mode='w+',
                #     shape=(N, 9, T, V, M))
                fp_sp = open_memmap(
                    save_name.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 6, T, V, M))

                for i in range(bch_len + 1):
                    if i % print_freq == 0:
                        print(f'{i} out of {bch_len}')
                    a_bch = torch.tensor(data[i * bch_sz:(i + 1) * bch_sz])

                    # generating bones
                    fp_sp_joint_list_bone = []

                    all_list = [
                        fp_sp_joint_list_bone
                    ]

                    for a_key in ntu_bone_angle_pairs:
                        a_angle_value = ntu_bone_angle_pairs[a_key]
                        a_bone_value = ntu_bone_adj[a_key]
                        the_joint = a_key - 1
                        a_adj = a_bone_value - 1
                        a_bch = a_bch.to('cuda')
                        bone_diff = (a_bch[:, :3, :, the_joint, :] -
                                     a_bch[:, :3, :, a_adj, :]).unsqueeze(3).cpu()
                        fp_sp_joint_list_bone.append(bone_diff)

                    for a_list_id in range(len(all_list)):
                        all_list[a_list_id] = torch.cat(all_list[a_list_id], dim=3)

                    all_list = torch.cat(all_list, dim=1)

                    # if fp_sp is None:
                    #     # fp_sp = all_list
                    #     fp_sp = all_list.numpy()
                    # else:
                    #     # fp_sp = torch.cat((fp_sp, all_list), dim=0)
                    #     fp_sp = np.concatenate((fp_sp, all_list), axis=0)

                    # Joint features.
                    fp_sp[i * bch_sz:(i + 1) * bch_sz, :3, :, :, :] = a_bch.cpu().numpy()
                    # Bone and other features.
                    fp_sp[i * bch_sz:(i + 1) * bch_sz, 3:, :, :, :] = all_list.numpy()

                print('fp sp: ', fp_sp.shape)
                # save_f_name = save_name.format(benchmark, part)
                # with open(save_f_name, 'wb') as f:
                #     np.save(f, fp_sp)
