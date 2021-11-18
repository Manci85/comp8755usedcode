import numpy as np
from numpy.lib.format import open_memmap

dataset = 'ntu120'
parts = {'train', 'val'}

benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    'ntu120': ('ntu120/xsub',),
    'kinetics': ('kinetics',)
}

save_name = '../data/{}/{}_data_joint_bone_angle_cangle_hands_2hands_2stream.npy'

for benchmark in benchmarks[dataset]:
    for part in parts:
        data_spatial = np.load('../data/{}/{}_data_joint_bone_angle_cangle_hands_2hands.npy'.format(benchmark, part), mmap_mode='r')
        data_velocity = np.load('../data/{}/{}_joint_bone_angle_cangle_hands_2hands_velocity.npy'.format(benchmark, part), mmap_mode='r')

        N, C, T, V, M = data_spatial.shape
        print('data shape: ', data_spatial.shape)
        fp_sp = open_memmap(
            save_name.format(benchmark, part),
            dtype='float32',
            mode='w+',
            shape=(N, C*2, T, V, M))
        print('fp_sp: ', fp_sp.shape)

        fp_sp[:, :C, :, :, :] = data_spatial
        fp_sp[:, C:2*C, :, :, :] = data_velocity
