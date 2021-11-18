import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

ntu_belly = 2

ntu_skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
)

# 相对于肚子的向量
ntu_skeleton_relative_to_center = []
for a_joint_id in range(1, 25+1):
    ntu_skeleton_relative_to_center.append((a_joint_id, ntu_belly))

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

relative_pairs = {
    'ntu/xview': ntu_skeleton_relative_to_center,
    'ntu/xsub': ntu_skeleton_relative_to_center,

    # NTU 120 uses the same skeleton structure as NTU 60
    'ntu120/xsub': ntu_skeleton_relative_to_center,
    'ntu120/xset': ntu_skeleton_relative_to_center,

    'kinetics': None,
}

benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    # 'ntu': ('ntu/xview',),
    'ntu120': ('ntu120/xset',),
    'kinetics': ('kinetics',)
}

parts = {'train', 'val'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NTU60/NTU120/Kinetics')
    parser.add_argument('--dataset', choices=['ntu', 'ntu120', 'kinetics'])
    parser.add_argument('--edge-type', choices=['bone', 'relative_belly'])
    args = parser.parse_args()
    args.dataset = 'ntu120'
    args.edge_type = 'bone'

    save_name = None

    if args.edge_type == 'bone':
        save_name = '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                    'data/{}/{}_data_bone.npy'
    else:
        raise NotImplementedError('Unsupported edge type. ')

    for benchmark in benchmarks[args.dataset]:
        for part in parts:
            print(benchmark, part)

            data = np.load('/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                           'data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
            N, C, T, V, M = data.shape
            print('data shape: ', data.shape)
            print('loading data from: ', '/data_seoul/zhenyue/Baseline-Implementation/Skeleton-Action-Recognition/2020-CVPR-Liu-MS-G3D/MS-G3D/' \
                           'data/{}/{}_data_joint.npy'.format(benchmark, part))
            print('saving to: ', save_name)
            fp_sp = open_memmap(
                save_name.format(benchmark, part),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, V, M))
            print('fp_sp: ', fp_sp.shape)

            fp_sp[:, :C, :, :, :] = data

            the_bone_pairs = None
            if args.edge_type == 'bone':
                the_bone_pairs = bone_pairs[benchmark]
            elif args.edge_type == 'relative_belly':
                the_bone_pairs = relative_pairs[benchmark]

            print('start --- ')
            for v1, v2 in tqdm(the_bone_pairs):
                if benchmark != 'kinetics':
                    v1 -= 1
                    v2 -= 1
                    fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
                else:
                    fp_sp[:, :2, :, v1, :] = data[:, :2, :, v1, :] - data[:, :2, :, v2, :]
                    fp_sp[:, 2, :, v1, :] = data[:, 2, :, v1, :] - data[:, 2, :, v2, :]

